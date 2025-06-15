from typing import Optional, Union
import json
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
import ray
import torch
from codetiming import Timer
import traceback
from tqdm import tqdm
import numpy as np
import signal
import time
import copy
import asyncio
import aiohttp
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import uuid

from roll.pipeline.rlvr.rlvr_config import RewardConfig
from roll.utils.local_code.evaluator import codegen_metrics
from roll.utils.local_code.extract_utils import extract_code_generation
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.logging import get_logger


def remove_entripoints(code, language="python"):
    if language == "python":
        if 'if __name__ == "__main__":' in code:
            next_line = code.index('if __name__ == "__main__":')
            code = code[:next_line].strip()
        elif "if __name__ == '__main__':" in code:
            next_line = code.index("if __name__ == '__main__':")
            code = code[:next_line].strip()
    elif language == "cpp":
        if "int main()" in code:
            next_line = code.index("int main()")
            code = code[:next_line].strip()
    elif language == "go":
        code = code.replace("package main", "")
    if "# Example usage" in code:
        next_line = code.index("# Example usage")
        code = code[:next_line].strip()

    if language == "python" and "def" in code:
        lines = code.strip().split("\n")
        cleaned_lines = [line for line in lines if line.startswith(" ") or line.startswith("def ")]
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        code = "\n".join(cleaned_lines)
        code = code + "\n"
    return code


def modified_text(text):
    def remove_spaces_after_newlines(text):
        return re.sub(r"(\n) +(?=\S|\n)", r"\1", text)

    def remove_extra_spaces(text):
        return re.sub(r" {2,}", " ", text)

    return remove_extra_spaces(remove_spaces_after_newlines(text)).strip().replace(" \n", "\n")


class CodeTester:
    def __init__(self, sandbox_url: str):
        self.DEFAULT_TIMEOUT = 10
        self.SOLUTION_IMPORTS = {
            "python": "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
        }
        self.sandbox_url = sandbox_url
        self.logger = get_logger()

    def check_format(self, prompt_id: str, text: str):
        # Check if format meets requirements: response must contain "</think>" and code blocks
        has_think_tag = "</think>" in text
        has_code_block = "```" in text

        format = 1 if has_think_tag and has_code_block else 0

        if not has_think_tag:
            self.logger.info(f"Response missing </think> tag: {prompt_id}")
        if not has_code_block:
            self.logger.info(f"Response missing code block: {prompt_id}")
        return format

    def extract_code_blocks(self, prompt, text: str, case_type: str = "input"):
        """Extract code blocks"""
        if "<|begin_of_solution|>" in text:
            text = text.split("<|begin_of_solution|>")[-1].strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if "```" in text:
            code_pattern = r"```(cpp|python|python3|java|javascript|ruby|go)\s*\n*(.*?)```"
            matches = re.findall(code_pattern, text, re.DOTALL)
            codes = []
            langs = []
            if matches:
                for lang, code in matches:
                    codes.append(code.strip())
                    langs.append(lang)
            if len(codes) == 0:
                code_pattern = r"```\s*\n*(.*?)```"
                matches = re.findall(code_pattern, text, re.DOTALL)
                codes = [match.strip() for match in matches]
        elif len(text) > 4000:
            return None, None, "No code block found"
        else:
            codes = [text.strip()]
            langs = ["python"] * len(codes)
        if codes is None or len(codes) == 0:
            return None, None, "No code block found"
        if case_type != "input" and "```python\ndef " in prompt:
            codes = [remove_entripoints(code, lang) for code, lang in zip(codes, langs)]
        return codes, langs, ""

    def format_sandbox_test(self, test_code, code_language, case_type, test_cases) -> Optional[List[Dict]]:
        """Format sandbox test cases"""
        test_cases_final = []
        if code_language is None or code_language == "":
            # TDO detect programming language
            code_language = "python"
        if len(test_cases) == 0:
            return None, "test case is empty"
        if case_type == "assert" or case_type == "pytest":
            if case_type == "pytest":
                code_language = "pytest"
            for case in test_cases:
                assert_code = case["assert_code"] if "assert_code" in case else case
                case_code = self.SOLUTION_IMPORTS.get(code_language, "") + test_code + "\n" + assert_code
                test_cases_final.append(
                    {
                        "compile_timeout": self.DEFAULT_TIMEOUT,
                        "run_timeout": self.DEFAULT_TIMEOUT,
                        "code": case_code,
                        "language": code_language,
                        "stdin": "",
                        "expected_stdout": "",
                    }
                )
        else:
            for test_case in test_cases:
                try:
                    test_cases_final.append(
                        {
                            "compile_timeout": self.DEFAULT_TIMEOUT,
                            "run_timeout": self.DEFAULT_TIMEOUT,
                            "code": self.SOLUTION_IMPORTS.get(code_language, "") + test_code,
                            "language": code_language,
                            "stdin": test_case["stdin"],
                            "expected_stdout": test_case["expected_stdout"],
                        }
                    )
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print(case)
        return test_cases_final, ""

    async def process_single_test(
        self, curid, session: aiohttp.ClientSession, test_case: Dict, max_retries: int = 3
    ) -> Tuple[Dict, Dict]:
        results = []
        for i in range(2):
            retries = 0
            result = None
            while retries < max_retries:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "eagleeye-traceid": str(uuid.uuid4()),
                    }
                    async with session.post(self.sandbox_url, headers=headers, json=test_case) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "status" in result and result["status"] == "SandboxError":
                                error_msg = result.get("message", "Unknown sandbox error")
                                self.logger.warning(f"curid: {curid}, Sandbox error: {error_msg}, retry: {retries+1}/{max_retries}")
                                retries += 1
                                await asyncio.sleep(1)
                                continue
                            else:
                                break
                        else:
                            error_text = await response.text()
                            self.logger.warning(
                                f"curid: {curid}, HTTP error {response.status}: {error_text}, retry: {retries+1}/{max_retries}"
                            )
                            retries += 1
                            await asyncio.sleep(1)
                except asyncio.TimeoutError:
                    self.logger.warning(f"curid: {curid}, Request timeout, retry: {retries+1}/{max_retries}")
                    retries += 1
                    await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"curid: {curid}, Sandbox error: {e}, retry: {retries+1}/{max_retries}")
                    retries += 1
                    await asyncio.sleep(1)
            if result:
                results.append(result)

        return test_case, results

    async def sandbox_test_async(self, prompt_id: str, test_cases: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        succeed_test_cases = []
        total_res = []
        test_count = len(test_cases)
        concurrency_limit = 20

        connector = aiohttp.TCPConnector(
            limit=concurrency_limit,
            ttl_dns_cache=600,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=60,
            ssl=False,
        )
        timeout = aiohttp.ClientTimeout(total=90, connect=20, sock_read=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(concurrency_limit)

            async def bounded_process_single_test(test_case):
                async with semaphore:
                    return await self.process_single_test(prompt_id, session, test_case)

            tasks = [bounded_process_single_test(test_case) for test_case in test_cases]

            for task in asyncio.as_completed(tasks):
                try:
                    test_case, results = await task
                    if test_case and results:
                        succeed_test_cases.append(test_case)
                        total_res.append(results)
                except Exception as e:
                    self.logger.error(f"prompt_id: {prompt_id}, Task error: {e}")
        return succeed_test_cases, total_res

    def sandbox_test(self, prompt_id: str, test_cases: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        result = asyncio.run(self.sandbox_test_async(prompt_id, test_cases))
        return result

    def sanbox_result_judge(self, test_cases: List[Dict], sandbox_results: List[Dict]) -> int:
        """Judge the number of test cases passed"""
        pass_test_number = 0
        error_types = []
        for i, responses in enumerate(sandbox_results):
            flag = "[No Pass]"
            for response in responses:
                status = response["status"]
                sanbox_output = ""
                if "expected_stdout" in test_cases[i]:
                    expected_output = test_cases[i].get("expected_stdout", "").strip()
                else:
                    expected_output = test_cases[i].get("expected_output", "").strip()
                if status == "Success":
                    sanbox_output = response.get("run_result", {}).get("stdout", "").strip()
                    if "User customization applied." in sanbox_output:
                        sanbox_output = sanbox_output.replace("User customization applied.", "").strip()
                    if "User customization module loaded!" in sanbox_output:
                        sanbox_output = sanbox_output.replace("User customization module loaded!", "").strip()
                    if "User" in sanbox_output:
                        if expected_output in sanbox_output:
                            flag = "[Pass]"
                    elif (
                        expected_output == ""
                        or sanbox_output == expected_output
                        or modified_text(sanbox_output) == modified_text(expected_output)
                    ):
                        flag = "[Pass]"
                    if flag == "[No Pass]":
                        error_types.append("LogicError")
                elif status == "Failed":
                    try:
                        stderr = response.get("run_result", {}).get("stderr", "").strip()
                        stderr = stderr.split("\n")[-1].split(":")[0].strip()
                        if stderr == "":
                            stderr = response.get("run_result", {}).get("return_code", "").strip()
                            error_types.append(f"ReturnCode: {stderr}")
                        else:
                            error_types.append(stderr)
                    except:
                        error_types.append("Others")
            if flag == "[Pass]":
                pass_test_number += 1
        error_types = list(set(error_types))
        return pass_test_number, error_types

    def single_code_test(
        self,
        global_step,
        prompt_id,
        code: str,
        case_type: str,
        test_cases: List[Dict],
        prompt: str = "",
        flag: int = 0,
    ):
        """Single code test"""
        info = {
            "global_step": global_step,
            "prompt_id": prompt_id,
            "pass_test_ratio": 0,
            "origin_response": code,
            "test_code": code,
            "test_cases_info": test_cases,
            "sanbox_responses": "",
            "succeed_test_cases_number": 0,
            "pass_test_number": 0,
            "validation": 1,
            "format_validation": 0,
            "error_info": [],
        }

        # Check if format meets requirements
        format_validation = self.check_format(prompt_id, code)
        info["format_validation"] = format_validation
        start_time = time.time()
        # Extract code blocks
        codes, code_langs, error_info = self.extract_code_blocks(prompt, code, case_type)
        if error_info != "" or len(codes) == 0:
            info["error_info"] = ["extract_code_blocks error"]
            return info

        test_code = codes[0]
        if len(code_langs) > 0:
            code_language = code_langs[0]
        else:
            # TDO detect programming language
            code_language = "python"

        # Format sandbox test cases
        test_cases, error_info = self.format_sandbox_test(test_code, code_language, case_type, test_cases)
        if error_info != "" or test_cases == None:
            info["error_info"] = ["format_sandbox_test error"]
            return info

        # Call sandbox testing
        succeed_test_cases, responses = self.sandbox_test(prompt_id, test_cases)
        if not responses or len(succeed_test_cases) == 0:
            info["error_info"] = ["sandbox error"]
            info["sanbox_responses"] = responses
            info["validation"] = 0
            return info

        # Judge sandbox test results
        pass_test_number, error_types = self.sanbox_result_judge(succeed_test_cases, responses)

        time_duration = time.time() - start_time
        self.logger.debug(
            f"prompt_id: {prompt_id}, total case number: {str(len(succeed_test_cases))} pass case number: {str(pass_test_number)} pass rate: {str(pass_test_number / len(succeed_test_cases))}, time: {time_duration}, detailed info: {error_info}"
        )

        pass_test_ratio = pass_test_number / len(succeed_test_cases) if len(succeed_test_cases) > 0 else 0
        info["test_code"] = test_code
        info["test_cases_info"] = test_cases
        info["sanbox_responses"] = responses
        info["succeed_test_cases_number"] = len(succeed_test_cases)
        info["pass_test_number"] = pass_test_number
        info["pass_test_ratio"] = pass_test_ratio
        info["error_info"] = error_types
        return info



def cal_http_sandbox(global_step, prompt_id, prompt, response, case_type, test_cases, url):
    codetester = CodeTester(url)
    info = codetester.single_code_test(global_step, prompt_id, response, case_type, test_cases, prompt)

    validation = info.get("validation", 0)
    pass_test_ratio = info.get("pass_test_ratio", 0)

    if validation == 0:
        return -1, info

    correct = 1 if pass_test_ratio >= 1 else 0
    return correct, info


def run_assert_tests(code, test_cases, timeout=20):
    """
    Run assert-style test cases against the provided code.
    """
    from roll.utils.local_code.execute_utils import BASE_IMPORTS, codeexecute_check_correctness
    
    all_passed = True
    for test_case in test_cases:
        assert_code = test_case["assert_code"]
        
        # Check if this is a assert-style test case (contains test_ functions)
        if "def test_" in assert_code:
            test_functions = []
            lines = assert_code.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("def test_"):
                    func_name = line.strip().split("(")[0].split("def ")[1]
                    test_functions.append(func_name)
            
            test_runner = "\n\n# Test runner\ntry:\n"
            for func_name in test_functions:
                test_runner += f"    {func_name}()\n"
            test_runner += "    print('All tests passed!')\nexcept AssertionError as e:\n    print(f'Test failed: {e}')\n    exit(1)"

            full_code = f"{BASE_IMPORTS}\n{code}\n{assert_code}\n{test_runner}"
        else:
            # Simple assert statement
            full_code = f"{BASE_IMPORTS}\n{code}\n{assert_code}"
        
        passed = codeexecute_check_correctness(full_code, timeout=timeout)
        if not passed:
            all_passed = False
            break
    
    return all_passed


def cal_local_test(prompt_id, response, test_cases, func_name=None, num_process_evaluate=4, timeout=20):
    """
    Reference implementation from: https://github.com/LiveCodeBench/LiveCodeBench/tree/main/lcb_runner
    
    Supports three testing modes:
    1. Input/output testing: Test cases contain stdin/expected_stdout pairs
    2. Assert testing: Test cases contain simple assert statements
    """
    extracted_code = ""
    info = {
        "prompt_id": prompt_id,
        "pass_test_ratio": 0,
        "origin_response": response,
        "test_code": response,
        "format_validation": 1,
        "validation": 1,
    }
    
    extracted_code = extract_code_generation(response)
    info["test_code"] = extracted_code
    
    is_assert_style = False
    is_pytest_style = False
    
    if isinstance(test_cases, list) and len(test_cases) > 0:
        if isinstance(test_cases[0], dict) and test_cases[0].get("assert_code", ""):
            is_assert_style = True
    
    correct = 0
    if is_assert_style:
        # Handle simple assert test cases
        all_passed = run_assert_tests(extracted_code, test_cases, timeout=timeout)
        if all_passed:
            info["pass_test_ratio"] = 1
            correct = 1
    else:
        # Handle traditional input/output test cases
        if func_name == "None":
            func_name = ""
            
        evaluation_sample = json.dumps(
            {
                "inputs": [t["stdin"] for t in test_cases],
                "outputs": [t["expected_stdout"] for t in test_cases],
                "fn_name": func_name,
            }
        )
        evaluation_sample = {"input_output": evaluation_sample}
        
        lcb_metrics, _, _ = codegen_metrics(
            [evaluation_sample],
            [[extracted_code]],
            k_list=[1],
            num_process_evaluate=num_process_evaluate,
            timeout=timeout,
            debug=False
        )
        
        if "pass@1" in lcb_metrics and lcb_metrics["pass@1"] == 100.0:
            info["pass_test_ratio"] = 1
            correct = 1
    
    return correct, info


class CodeSandboxRewardWorker(Worker):
    """
    (x)Reward Model uses AutoModelForSequenceClassification protocol
    Code-oriented sandbox unit test reward model
    """

    def __init__(self, worker_config: RewardConfig):
        super().__init__(worker_config=worker_config)
        self.worker_config = worker_config
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        self.use_local = self.worker_config.use_local
        self.url = self.worker_config.code_url

        self.max_resp_len = 10000

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_rewards(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'rewards': rewards})
        """
        global_step = data.meta_info.get("global_step", 0)
        verify_answer = []

        prompts_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        cur_force_lcb = False
        for i, (prompt_id, prompt_txt, response, case_type, test_cases, test_case_function, tag) in enumerate(
            zip(
                data.non_tensor_batch["id"],
                prompts_text_list,
                response_text_list,
                data.non_tensor_batch["case_type"],
                data.non_tensor_batch["test_cases"],
                data.non_tensor_batch["test_case_function"],
                data.non_tensor_batch["tag"],
            )
        ):
            if "livecodebench" in tag:
                cur_force_lcb = True
            if isinstance(test_cases, str):
                test_cases = json.loads(test_cases)
            if cur_force_lcb or self.use_local:
                correct, info = cal_local_test(prompt_id, response, test_cases, test_case_function)
            else:
                correct, info = cal_http_sandbox(
                    global_step, prompt_id, prompt_txt, response, case_type, test_cases, self.url
                )

            self.logger.debug(f"{json.dumps(info, ensure_ascii=False)}")

            verify_answer.append(correct)
            
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = torch.tensor(verify_answer, dtype=torch.float16)
        scores = torch.tensor(verify_answer, dtype=torch.float16)

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores,
            }
        )

        success_count = (scores == 1).sum().item()
        fail_count = (scores == 0).sum().item()
        invalid_count = (scores == -1).sum().item()
        total_count = len(scores)

        self.logger.info(
            f"Batch results - Total: {total_count}, Success: {success_count}, "
            f"Fail: {fail_count}, Invalid: {invalid_count}"
        )

        return output
