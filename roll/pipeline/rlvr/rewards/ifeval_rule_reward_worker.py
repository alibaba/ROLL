# 导入必要的库和模块
from functools import partial
from typing import Optional, Union, Iterator
import json
import re
import inspect
import os

import ray
import torch
from codetiming import Timer
from tqdm import tqdm
import signal
import multiprocessing
import itertools, collections
from collections import defaultdict

# 从已有的 WorkerConfig、Worker、Dispatch 等模块导入
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
import string
from difflib import SequenceMatcher
import nltk

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
os.environ["NLTK_DATA"] = os.path.join(os.path.dirname(__file__), "nltk_data")
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

# 假设 tokenizer 依然来自 default_tokenizer_provider
from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider

# 引入 ifeval 验证函数的字典映射
# IF_FUNCTIONS_MAP 是题主在上面给出的完整实现中包含的函数映射
from typing import Union, Dict, List

from roll.utils.logging import get_logger

logger = get_logger()  # 获取日志记录器实例


def first_boxed(text: str) -> str | None:
    """
    提取第一个 \boxed{...} 的内容，支持 boxed 内部再嵌套 {}。
    """
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return ""  # 没找到 \boxed{

    i = start + len(marker)  # 跳过 '\boxed{'
    depth = 1  # 已进入 1 层 {
    buf = []

    while i < len(text) and depth:  # 扫描直到配平
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:  # 恢复到 0 说明 boxed 完成
                break
        if depth:  # 只在括号未配平时记录字符
            buf.append(ch)
        i += 1

    return "".join(buf) if depth == 0 else ""


class timeout:
    """
    与 MathRewardWorker 示例中类似的超时上下文，用于演示，
    如果不需要超时，可直接省略。
    """

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# 包含关键字：在你的回答中应包含关键字 {keyword1}、{keyword2}
def verify_keywords(text, keyword_list):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 将响应文本转换为小写，以便进行不区分大小写的匹配
    response_lower = text.lower()

    # 检查响应中是否包含所有关键字（每个关键字也转换为小写进行匹配）
    return all(keyword.lower() in response_lower for keyword in keyword_list)


# 关键字出现频率：在你的回答中，单词 {word} 应该出现 {N} 次
def verify_keyword_frequency(text, word, N):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 将文本转换为小写，使搜索不区分大小写
    text = text.lower()
    word = word.lower()

    # 使用正则表达式匹配单词边界，将文本切分为单词列表
    words = re.findall(r"\b\w+\b", text)

    # 统计实际出现次数（精确匹配关键字）
    actual_count = sum(1 for word in words if word == word)

    # 检查实际出现次数是否等于期望的 N 次
    constraint_met = actual_count == N

    return constraint_met


# 禁止出现特定单词：回答中不应包含关键字 {forbidden words}
def validate_forbidden_words(text, forbidden_words):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 将文本转换为小写，进行不区分大小写的匹配
    text_lower = text.lower()

    # 检查每个禁止单词是否出现在文本中
    found_words = [word for word in forbidden_words if word.lower() in text_lower]

    # 如果没有找到禁止单词，返回 True；否则返回 False
    return len(found_words) == 0


# 字母出现频率：在你的回答中，字母 {letter} 应该恰好出现 {N} 次
def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if len(letter) != 1:
        raise ValueError("字母参数必须为单个字符")

    actual_count = text.count(letter)
    return actual_count == N


# 回答语言约束：你的整个回答应当使用 {language}，不允许包含其他语言内容
def validate_response_language(text, language):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    from langdetect import detect

    # 检测文本的语言
    detected_language = detect(text)
    # 检查检测到的语言是否与预期语言相符
    return detected_language == language


# 段落数量：回答中应包含 {N} 个段落，段落之间使用 markdown 分隔符 "* * *" 隔开
def verify_paragraph_count(text: str, N: int) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """

    def clean_text(text: str) -> str:
        """移除多余空白字符，并规范换行符"""
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    # 清理输入文本
    text = clean_text(text)

    # 依据 markdown 分隔符分割文本，每个分隔符会创建 n+1 个段落
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)

    # 验证每个分割结果中是否包含非空内容
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False

    return actual_count == N


# 单词数量约束：回答中的单词数应至少/大约/最多达到 {N} 个
def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 清除多余空白字符并拆分文本为单词列表
    words = text.strip().split()
    actual_count = len(words)

    # 定义 "around" 约束的容错范围（目标单词数的 ±10%，至少 1 个单词）
    tolerance = max(round(N * 0.1), 1)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


# 句子数量约束：回答中应包含至少/大约/最多 {N} 个句子
def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 使用正则表达式根据句号或问号后的空格拆分文本为句子列表
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # 统计实际句子数量
    actual_count = len(sentences)

    # 根据不同的量词进行比较
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


# 段落数量及指定段落首词约束：回答中应包含 {N} 个段落，段落之间仅以两个换行符分隔，第 {i} 个段落必须以 {first word} 开头
def validate_paragraphs(text, N, first_word, i):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 根据两个换行符分割文本为段落
    paragraphs = text.split("\n\n")

    # 检查段落总数是否符合要求
    if len(paragraphs) != N:
        return False

    # 检查第 i 个段落的开头是否为指定单词
    if paragraphs[i - 1].strip().startswith(first_word):
        return True
    return False


# 附言验证：请在回答末尾明确添加以 {postscript marker} 开头的附言
def verify_postscript(text, postscript_marker):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 检查文本中是否包含附言标记
    if postscript_marker in text:
        # 获取标记的索引位置
        marker_index = text.find(postscript_marker)
        # 检查标记附近是否还有其它内容
        remaining_text = text[marker_index:].strip()
        # 验证附言不只是标记本身而已
        return len(remaining_text) > len(postscript_marker)
    return False


# 占位符验证：回答中应至少包含 {N} 个用方括号表示的占位符，例如 [address]
def validate_placeholders(text: str, N: int) -> tuple[bool, List[str]]:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 使用正则表达式查找所有位于方括号内的内容
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)

    # 检查是否至少找到了 N 个占位符
    has_enough = len(placeholders) >= N

    return has_enough


# 项目符号验证：回答必须包含恰好 {N} 个项目符号点。请使用 markdown 格式的项目点，例如：* 这是一个点。
def verify_bullet_points(text: str, N: int) -> tuple[bool, str]:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # 按行拆分文本，并统计以 * 或 - 开头的行
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    actual_count = len(bullet_points)

    if actual_count == N:
        return True
    else:
        return False


# 标题验证：回答中必须包含一个标题，用双尖括号包裹，例如 <<poem of joy>>
def validate_title(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)

    if len(matches) > 0:
        return True
    else:
        return False


# 选择题验证：回答内容必须为以下选项之一：{options}
def validate_choice(text: str, options: list) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    for option in options:
        if text in option:
            return True
    return False


# 高亮区域数量验证：回答中必须至少高亮 {N} 个区域，使用 markdown 格式，比如 *highlighted section*
def validate_highlighted_sections(text: str, N: int) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)

    if len(matches) >= N:
        return True
    else:
        return False


# 多区块验证：回答中必须包含 {N} 个区块，每个区块的开始都应以 {section splitter} 开头
def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    sections = text.split(section_splitter)
    # 第一个区块可能不以分割符开头，因此需要做调整
    if sections[0] == "":
        sections.pop(0)
    if len(sections) == N:
        return True
    else:
        return False


# JSON 格式验证：整个输出必须使用 JSON 格式包裹
def validate_json_format(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


# 重复提示验证：首先重复用户的请求内容不做更改，然后再给出你的回答（重复内容不应包含其他额外信息）
def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.startswith(original_prompt):
        return True
    else:
        return False


# 两种回答验证：提供两种不同的回答，两个回答之间仅用六个星号 "******" 分隔开
def validate_two_responses(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


# 全部大写：整个回答必须全部使用英文大写字母
def validate_uppercase(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text == text.upper():
        return True
    else:
        return False


# 全部小写：整个回答必须全部使用英文小写字母，不允许有大写字母
def validate_lowercase(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text == text.lower():
        return True
    else:
        return False


# 全大写单词出现频率验证：在回答中，全大写单词的出现次数应满足至少/大约/最多 {N} 次
def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return len(words) == N
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


# 结束语验证：回答最后必须以确切的短语 {end phrase} 结束，且该短语后面不允许有其他内容
def validate_end(text: str, end_phrase: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.endswith(end_phrase):
        return True
    else:
        return False


# 引号包装验证：整个回答必须用双引号包裹起来
def validate_quotation(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.startswith('"') and text.endswith('"'):
        return True
    else:
        return False


# 禁用逗号：整个回答中不允许出现任何逗号
def validate_no_commas(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if "," not in text:
        return True
    else:
        return False


def call_ifeval_function(func, text: str, constraint_dict: dict):
    """
    1) 获取func的函数签名
    2) 只保留与签名匹配且非None的参数
    3) 调用func(text, **filtered_args)
    """
    # 1) 获取函数签名
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())  # 该函数形参名的集合

    # 2) 过滤掉 constraint_dict 中的无关字段和 None 值
    #    （如果一个函数的参数刚好是 None 值也是合法，就保留；否则你可以额外判断）
    filtered_args = {}
    for k, v in constraint_dict.items():
        if k in valid_params:  # 形参里确实有这个字段
            # 如果你想彻底丢弃 None，也可以加上: if v is not None:
            filtered_args[k] = v

    # 3) 调用函数
    return func(text, **filtered_args)


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    """
    # 如果 max_penalty 是正的，这里直接抛出错误，说明要用负值来做惩罚
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    # 内部函数 zipngram，用于切分文本为 ngram
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    # repetition_penalty_reward 函数用于计算在给定 response 中，n-gram 的重复程度
    def repetition_penalty_reward(response, **kwargs) -> float:
        """
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
        """
        # 如果回复为空或不足 ngram 大小，则直接返回 0
        if response == "" or len(response.split()) < ngram_size:
            return 0.0

        # 遍历所有 ngram，统计 unique ngram 和 total ngram 的数量
        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1

        # scaling = 1 - (不重复的 ngram / 总的 ngram 数量)
        # 不重复的越少（重复越多）scaling 越大
        scaling = 1 - len(ngrams) / total
        # reward 是 scaling 乘以 max_penalty
        reward = scaling * max_penalty
        return reward

    return repetition_penalty_reward


def extract_after_last_think(input_string, end_think="</think>"):
    """
    提取输入字符串中最后一个 '</think>' 标签之后的内容，
    并移除结果字符串开头的所有换行符。

    Args:
    input_string: 原始字符串。

    Returns:
    提取并处理后的字符串。如果未找到 '</think>' 标签，则返回空字符串。
    """
    # 查找最后一个 end_think 的起始位置
    last_index = input_string.rfind(end_think)

    # 如果没有找到 end_think
    if last_index == -1:
        # return ""
        return input_string  # 或者根据需要返回 None 或原始字符串

    # 计算 end_think 结束后的位置
    start_pos = last_index + len(end_think)

    # 提取 end_think 之后的部分
    extracted_part = input_string[start_pos:]

    # 移除开头的所有换行符 '\n'
    cleaned_part = extracted_part.lstrip("\n")

    return cleaned_part


class GeneralRuleRewardWorker(Worker):
    """
    一个示例 Reward Worker，用于执行 ifeval 验证并把每个 func 的结果放到 output.tensors 中。
    在此示例里，ground_truths的str
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5)
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')
        # use os to export a nltk_data

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        """
        仅调用 data.non_tensor_batch['ground_truth'] 中的 “func_name”，
        并将其结果作为单一的 response-level 奖励返回。
        """

        # 1) 解码回复文本
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
        batch_size = len(response_text_list)

        # 2) 读取 ground_truth（其中是一串 JSON，包含 func_name 等参数）
        prompts = data.non_tensor_batch["prompt"]
        ground_truths = data.non_tensor_batch["ground_truth"]
        tags = data.non_tensor_batch["tag"]

        # 3) 准备一个列表存放验证结果
        results = [0.0] * batch_size
        repetition_penalty_rewards = []
        response_length_rewards = []

        for i, (resp_tokens, ground_truth, tag, prompt) in enumerate(
            zip(data.batch["responses"], ground_truths, tags, prompts)
        ):
            # 解码当前条目
            resp_text = self.tokenizer.decode(resp_tokens, skip_special_tokens=False)
            resp_text1 = resp_text.replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "")
            resp_text = extract_after_last_think(resp_text1)
            # logger.info(f"extract_after_last_think(resp_text): {resp_text}")

            if tag == "ifeval":
                # 解析 ground_truth (JSON) 得到约束信息
                if isinstance(ground_truth, str):
                    constraint_dict = json.loads(ground_truth)
                else:
                    constraint_dict = ground_truth  # 如果已经是 dict，就直接用

                # 从约束中取出 func_name
                func_name = constraint_dict.get("func_name", None)
                if not func_name or func_name not in IF_FUNCTIONS_MAP:
                    self.logger.warning("constraint missing func_name")
                    # 如果无 func_name 或没找到对应函数
                    # 那么这里我们将结果记为 0.0（也可做别的处理）
                    results[i] = 0.0
                    continue

                # 移除 func_name，其它参数传给函数
                constraint_dict.pop("func_name")
                func = IF_FUNCTIONS_MAP[func_name]
                # print(f"Running function {func_name} with Response text: {resp_text}")
                # print(f"Response text: {resp_text}")

                # 调用函数进行验证
                try:
                    result = call_ifeval_function(func, resp_text, constraint_dict)
                except Exception as e:
                    self.logger.error(f"Error in function {func_name}: {e}")
                    result = False
            else:
                self.logger.warning(f"Unknown tag: {tag}")

            # 将结果转为 float: bool -> (1.0/0.0), 数值 -> float(...), 其他结构 -> bool(...)
            if isinstance(result, bool):
                val = 1.0 if result else 0.0
            elif isinstance(result, (int, float)):
                val = float(result)
            else:
                val = 1.0 if result else 0.0

            # 存到 results
            results[i] = val
            repetition_penalty_rewards.append(self.repetition_penalty_reward_fn(resp_text1))

        # 4) 准备输出张量：
        #   - token_level_rewards：形状与 responses 相同、全 0
        #   - response_level_rewards：即 results
        #   - scores：可与 response_level_rewards 相同（用于统计/日志）
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        scores = torch.tensor(results, dtype=torch.float16)
        repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        response_level_rewards = scores + repetition_penalty_rewards

        # 5) 将这些张量打包进同一个字典
        output_tensors = {
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
            "scores": scores
        }

        # 6) 用 DataProto.from_dict(...) 构造返回值
        output = DataProto.from_dict(tensors=output_tensors)
        return output


IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

ALL_FUNCS = list(IF_FUNCTIONS_MAP.keys())
