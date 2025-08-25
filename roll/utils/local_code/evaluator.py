import ast
import json
import multiprocessing
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from roll.utils.local_code.execute_utils import BASE_IMPORTS, codeexecute_check_correctness
from roll.utils.local_code.pass_k_utils import compute_metrics_from_results

def codegen_check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.

    The global timeout is to catch some extreme/rare cases not handled by the
    timeouts inside `run_test`
    """

    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        from .testing_util import run_test
        try:
            with open(os.devnull, 'w') as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    res, metadata = run_test(sample=sample, test=generation, debug=debug, timeout=timeout)
                    result.append(res)
                    metadata_list.append(metadata)
                except Exception as e:
                    traceback.print_exc(10)
                    result.append([-1 for i in range(len(sample['inputs']))])
                    metadata_list.append({})
        except Exception as e:
            traceback.print_exc(10)

    with multiprocessing.Manager() as manager:
        result = manager.list()
        metadata_list = manager.list()
        p = multiprocessing.Process(
            target=_temp_run,
            args=(sample, generation, debug, result, metadata_list, timeout),
        )
    
        p.start()
        try:
            max_timeout = min(timeout * len(sample["inputs"]) + 1, 360)
            p.join(timeout=max_timeout)
        except Exception as e:
            if debug:
                print(f"Exception during process join: {repr(e)}")
            traceback.print_exc(10)
        finally:
            if p.is_alive():
                p.kill()
            p.join()
        if not result:
            result = [[-1 for i in range(len(sample["inputs"]))]]
            if debug:
                print("global timeout")
        if not metadata_list:
            metadata_list = [{}]
        return result[0], metadata_list[0]


def evaluate_generations_by_problem(problem_generations: list, sample: list, debug: bool, timeout: int):
    """Evaluate each problem.

    Args:
        problem_generations:
        sample:
        debug:
        timeout
    """
    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = codegen_check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(
                        f"Results were not True for all test cases"  # noqa: F541, E501
                        f" {curr_res=}\n"
                    )
        except Exception as e:
            if debug:
                print(
                    f"Compilation failed, test framework exception"  # noqa: F541, E501
                    f" = {repr(e)}{e}\n"
                )
            # break
            curr_metadata = {}
        finally:
            assert isinstance(curr_res, list)
            assert isinstance(curr_metadata, dict)
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            print(f"Sample\n{r}\nResult\n{res[i]}\nMetadata\n{metadata[i]}")
            print("*" * 30 + "\n\n")
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: List[List[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=60,
):
    """We take the list of code generations and try to compile them and the run
    their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS
            dataset)
        level: difficulty level used in the generation, can be "all",
            "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is
            a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test
            case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset
    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
        futures = {
            executor.submit(evaluate_generations_by_problem, problem_generations, sample, debug, timeout): index
            for (problem_generations, sample, debug, timeout), index in inputs
        }

        results = {}
        metadata = {}
        for future in as_completed(futures):
            index = futures[future]
            results[index], metadata[index] = future.result()

    assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)} {results=}"

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=60,
    debug=True,
):
    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(generations_list[0]), f"{len(final_metadata[i])=}"

    return [metrics, results, final_metadata]


def evaluate_score(args) -> List[bool]:
    gs, (c, i, o) = args

    execution_results = []
    for g in gs:
        if i in g:
            pass
        else:
            code_to_execute = f"{BASE_IMPORTS}\n{c}\nassert {o} == {g}"
            execution_results.append(codeexecute_check_correctness(code_to_execute, 3))
    if len(execution_results) == 0:
        execution_results = [False] * len(gs)
    return execution_results


def parse_assert_statement(statement):
    """Parse a Python assert statement and extract the expected output from the
    right side of the '==' operator as a string.

    :param statement: A string containing the assert statement.
    :return: The expected output from the assert statement as a string.
    """
    try:
        parsed = ast.parse(statement, mode="exec")
    except SyntaxError:
        return "Invalid syntax"

    if len(parsed.body) == 0:
        return "Empty statement"

    if not isinstance(parsed.body[0], ast.Assert):
        return "Not an assert statement"

    comparison = parsed.body[0].test

    if not isinstance(comparison, ast.Compare) or not isinstance(comparison.ops[0], ast.Eq):
        return "Not an equality assertion"

    # Extract and return the right side of the '==' operator as a string
    return ast.get_source_segment(statement, comparison.comparators[0])


def check_testcase_output(testcase_str, expected_output):

    if len(testcase_str.splitlines()) > 1:
        for line in testcase_str.splitlines():
            if line.startswith("#"):
                continue
            if "assert" in line:
                testcase_str = line
                break

    testcase_str = testcase_str.strip()

    if "assert" in testcase_str:
        testcase_output_str = str(parse_assert_statement(testcase_str))

    else:
        testcase_output_str = testcase_str

    global_result = None

    try:
        testcase_output_eval = eval(testcase_output_str)
    except Exception as e:
        print(e)
        global_result = False
        # print("Failed to eval testcase output", testcase_output_str)
        # breakpoint()

    try:
        expected_output_eval = json.loads(expected_output)
    except Exception as e:
        print(e)
        global_result = False
        print("Failed to eval expected testcase output", expected_output)

    if global_result is None:
        global_result = testcase_output_eval == expected_output_eval

    return global_result


def test_output_metrics(
    samples,
    generations,
    k_list=[1, 5],
    debug=False,
):
    num_samples = len(samples)
    results = []
    for idx in tqdm(list(range(num_samples))):
        idx_results = []
        sample = samples[idx]
        extracted_generation_list = generations[idx]
        for extracted_generation in extracted_generation_list:
            global_result = check_testcase_output(extracted_generation, sample["output"])
            idx_results.append([global_result])
        results.append(idx_results)

    results = {result_idx: results[result_idx] for result_idx in range(len(results))}

    metrics = compute_metrics_from_results(results, k_list=k_list)

    return [metrics, results]
