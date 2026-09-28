"""
Geo3K multi-turn VLM environment for agentic RL.

The model sees a geometry problem with an image and can call ``calc_score`` for
feedback. Reward is binary. Tool-call attempts continue until the answer is
correct or ``max_steps`` is reached; a response without a tool call is final.
"""
from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Any, SupportsFloat

import numpy as np
import ray
from dacite import from_dict
from gem import Env

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None

from PIL import Image

from .math_utils import grade_answer_verl

from roll.configs.data_args import DataArguments
from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.utils.qwen_vl_utils import fetch_image

logger = get_logger()

# Matches <tool_call>{...}</tool_call> payloads emitted by the model.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_geo3k_reward"}

TERMINAL_OBS = ""


def _grade_answer(answer: str, ground_truth: str) -> bool:
    """
    Grade ``answer`` against ``ground_truth`` using ``grade_answer_verl``.

    Accepts both boxed and raw strings by retrying with a ``\boxed{}`` wrapper.
    """
    answer = answer.strip()
    if not answer:
        return False
    candidates = [answer]
    if "\\boxed" not in answer:
        candidates.append(f"\\boxed{{{answer}}}")

    for candidate in candidates:
        try:
            if grade_answer_verl(candidate, ground_truth):
                return True
        except Exception:
            continue
    return False


def _grade_answer_safe(answer: str, ground_truth: str, timeout_sec: float = 5.0) -> bool:
    """Grade answer against ground truth, with error handling.

    Uses ``grade_answer_verl`` (sympy-based, no signal dependency) directly in
    the main process — safe in Ray worker threads unlike ``math_verify``.
    """
    try:
        return _grade_answer(answer, ground_truth)
    except Exception:
        return False


class Geo3kEnv(Env):
    """
    Multi-turn VLM environment for geo3k geometry problems.

    Flow:
      1. ``reset`` — sample a problem + image, return as VLM observation.
      2. Model reasons and emits ``<tool_call>{"name":"calc_score","arguments":{"answer":"..."}}</tool_call>``.
      3. ``step`` — grade tool-call answers and return feedback, or treat a
         response without a tool call as the final answer.
    """

    image_placeholder: str = "<image>"

    def __init__(
        self,
        data_args: dict | DataArguments,
        max_steps: int = 3,
        seed: int | None = None,
        mode: str = "train",
        question_key: str = "problem",
        answer_key: str = "answer",
        image_key: str = "images",
        image_max_pixels: int = 1280 * 28 * 28,
        format_penalty: float = 0.0,
        **_,
    ):
        Env.__init__(self)
        self.mode = mode
        self.max_steps = max_steps
        self.question_key = question_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.image_max_pixels = image_max_pixels
        self.format_penalty = format_penalty

        data_args = from_dict(data_class=DataArguments, data=data_args) if not isinstance(data_args, DataArguments) else data_args
        dataset_name = data_args.file_name

        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(
            name=f"{self.mode}_geo3k",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote(
            dataset_name=dataset_name,
            split="train",
            mode=global_dataset_mode,
            dataset_kwargs={"num_proc": data_args.preprocessing_num_workers},
        )
        self.dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name="geo3k", dataset_ref=self.dataset))

        # Episode state
        self.step_count = 0
        self.problem: str = ""
        self.answer: str = ""
        self.images: list[Image.Image] = []

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Sample a geometry problem and return it as a VLM observation."""
        Env.reset(self, seed)
        data: dict | None = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None

        self.problem = str(data[self.question_key])
        self.answer = str(data[self.answer_key])
        raw_images = data.get(self.image_key, [])
        if not isinstance(raw_images, (list, tuple)):
            raw_images = [raw_images]
        self.images = []
        for image in raw_images:
            try:
                if isinstance(image, dict):
                    image = image.get("bytes") or image.get("path")
                if isinstance(image, (bytes, bytearray)):
                    image = Image.open(BytesIO(image))
                self.images.append(fetch_image({"image": image, "max_pixels": self.image_max_pixels}))
            except Exception as exc:
                logger.warning("Failed to decode geo3k image: %s", exc)
                self.images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
        self.step_count = 0

        # Ensure the prompt contains one image placeholder for the VLM.
        problem_text = self.problem
        if self.image_placeholder not in problem_text:
            problem_text = f"{self.image_placeholder}\n{problem_text}"

        first_obs = {"prompt": problem_text, "image": self.images}
        return first_obs, {"env_instruction": ""}

    def step(
        self, action: str | Any,
    ) -> tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Parse the model's response, optionally score via calc_score, and return feedback.

        Returns ``(observation, reward, terminated, truncated, info)``.
        ``observation`` is a plain string (tool feedback) for subsequent turns.
        """
        self.step_count += 1

        # Handle non-string actions (e.g. EpisodeStopReason.MAX_LENGTH).
        if not isinstance(action, str):
            reward = 0.0
            info = self._build_info(raw_reward=reward, success=False, action_is_valid=False)
            return TERMINAL_OBS, reward, True, True, info

        response_text = action
        is_final_turn = self.step_count >= self.max_steps
        tool_call = self._extract_tool_call(response_text)

        # Without a tool call, treat the response as a final answer.
        if tool_call is None:
            answer_text = self._extract_answer_from_text(response_text)
            score = 1.0 if (answer_text and _grade_answer_safe(answer_text, self.answer)) else 0.0
            reward = score
            info = self._build_info(
                raw_reward=reward,
                success=score > 0,
                action_is_valid=answer_text is not None,
                tool_executed=False,
                answer=answer_text or "",
                score=score,
            )
            return TERMINAL_OBS, reward, True, score == 0.0, info

        name = (tool_call.get("name") or "").strip()
        arguments = tool_call.get("arguments") or {}

        if name not in SUPPORTED_TOOL_NAMES:
            obs = (
                f"Tool `{name}` is not supported. "
                'Call `calc_score` via <tool_call>{"name": "calc_score", "arguments": {"answer": "<your answer>"}}</tool_call> '
                "to check your solution."
            )
            info = self._build_info(tool_executed=False, action_is_valid=False)
            return obs, 0.0, is_final_turn, is_final_turn, info

        raw_answer = arguments.get("answer")
        parsed_answer = "" if raw_answer is None else str(raw_answer)

        if not parsed_answer.strip():
            obs = (
                "Tool call detected but no `answer` was provided. "
                'Call `calc_score` via <tool_call>{"name": "calc_score", "arguments": {"answer": "<your answer>"}}</tool_call> '
                "to check your solution."
            )
            info = self._build_info(tool_executed=False, action_is_valid=False, answer_missing=True)
            return obs, 0.0, is_final_turn, is_final_turn, info

        is_correct = _grade_answer_safe(parsed_answer, self.answer)
        score = 1.0 if is_correct else 0.0
        reward = score
        terminated = is_correct or is_final_turn
        truncated = is_final_turn and not is_correct

        obs = self._build_tool_feedback(score, parsed_answer)
        info = self._build_info(
            raw_reward=reward,
            success=is_correct,
            action_is_valid=True,
            tool_executed=True,
            answer=parsed_answer,
            score=score,
        )
        return obs, reward, terminated, truncated, info

    def _build_info(
        self,
        raw_reward: float = 0.0,
        success: bool = False,
        action_is_valid: bool = True,
        tool_executed: bool = False,
        answer: str = "",
        score: float | None = None,
        answer_missing: bool = False,
        **extra,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "action_is_valid": int(action_is_valid),
            "success": int(success),
            "raw_reward": raw_reward,
            "tool_call": int(tool_executed),
        }
        if score is not None:
            metrics["tool_score"] = score
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "raw_reward": "last",
            "tool_call": "mean",
            "tool_score": "last",
        }
        info: dict[str, Any] = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "tool_executed": tool_executed,
            "answer": answer,
            "score": score if score is not None else raw_reward,
        }
        if answer_missing:
            info["answer_missing"] = True
        info.update(extra)
        return info

    @staticmethod
    def _extract_tool_call(text: str) -> dict[str, Any] | None:
        """Parse the latest ``<tool_call>{...}</tool_call>`` payload."""
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None
        raw_json = matches[-1].group(1).strip()
        loader = orjson.loads if orjson is not None else json.loads
        try:
            payload = loader(raw_json)
        except Exception as exc:
            logger.warning("Failed to decode tool call payload: %s", exc)
            return None
        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Tool call arguments are not valid JSON; rejecting.")
                return None
        if not name:
            return None
        return {"name": name, "arguments": arguments}

    @staticmethod
    def _extract_answer_from_text(text: str) -> str | None:
        """Prefer the last ``\\boxed{}`` chunk; fall back to the last non-empty line."""
        last_boxed = text.rfind("\\boxed{")
        if last_boxed != -1:
            start = last_boxed + len("\\boxed{")
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == "\\" and not escaped:
                    escaped = True
                    continue
                if ch == '"' and not escaped:
                    in_string = not in_string
                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        if depth == 0:
                            return text[start:idx].strip()
                        depth -= 1
                escaped = False
        for line in reversed(text.splitlines()):
            cleaned = line.strip()
            if cleaned:
                return cleaned[:512]
        trimmed = text.strip()
        return trimmed[:512] if trimmed else None

    def _build_tool_feedback(self, score: float, parsed_answer: str) -> str:
        """Generate concise feedback for the model."""
        turn_idx = self.step_count - 1  # zero-based
        last_warning_turn = self.max_steps - 2 if self.max_steps >= 2 else self.max_steps - 1
        is_final_warning = turn_idx >= last_warning_turn

        if score == 1.0:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' matches the reference. "
                "You can now stop reasoning and provide the final solution in \\boxed{}."
            )
        if is_final_warning:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary. "
                "Since you only have one chance to answer, don't call tool again. You should provide your final answer in the form below Answer: \\boxed{$Answer} where $Answer is your final answer to this problem."
            )
        return (
            f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
            "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary."
        )

    def add_extra_data(self, data, messages):
        """Attach ground-truth metadata to the rollout."""
        from roll.distributed.scheduler.protocol import DataProto  # local import to avoid cycles

        if isinstance(data, DataProto):
            data.non_tensor_batch.update(
                {
                    "ground_truth": np.array([self.answer], dtype=object),
                    "problem": np.array([self.problem], dtype=object),
                }
            )
