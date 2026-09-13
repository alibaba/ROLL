"""
Message tracking for agentic training with prefix aggregation.

Manages incremental tokenization, hash-chain message deduplication, fork detection,
and traj-wise training data generation.
"""

import json
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from pydantic import TypeAdapter
from transformers import PreTrainedTokenizer

from sglang.srt.entrypoints.openai.protocol import ChatCompletionMessageParam

from roll.pipeline.agentic.env_manager.token_mask_utils import convert_list_content_str

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('ROLL_LOGGING_LEVEL', 'WARN'))

DEFAULT_HASH = "default_hash"

ChatCompletionMessageAdapter = TypeAdapter(ChatCompletionMessageParam)

# Dummy prefix for single-message tokenization
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


@dataclass
class MessageItem:
    """A single message with its tokenized representation."""
    index: int
    message: dict
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    # Multimodal placeholders
    image_data: Optional[Any] = None
    video_data: Optional[Any] = None
    audio_data: Optional[Any] = None
    modalities: Optional[List[str]] = None

    @property
    def role(self) -> str:
        return self.message["role"]


# ========================= Message Unification =========================


def unify_content(msg: dict) -> None:
    """Normalize content to list-of-dicts format with merged text."""
    if msg["content"] is None or msg["content"] == "":
        msg["content"] = []
    elif isinstance(msg["content"], str):
        text = msg["content"]
        msg["content"] = [{"type": "text", "text": text}]

    # Merge consecutive text items
    content: List[dict] = []
    text = ""
    has_text = False
    for item in msg["content"]:
        if item["type"] == "text":
            text += item["text"]
            has_text = True
        else:
            content.append(item)
    if has_text:
        content.append({"type": "text", "text": text})
    msg["content"] = content


def unify_tool_calls(msg: dict, tool_calls_dict: Dict[str, dict]) -> None:
    """Replace tool_calls with canonical versions from tool_calls_dict."""
    if (
        msg["role"] == "assistant"
        and "tool_calls" in msg
        and isinstance(msg["tool_calls"], list)
    ):
        unified = []
        for item in msg["tool_calls"]:
            canonical = tool_calls_dict[item["id"]]
            unified.append(canonical)
        msg["tool_calls"] = unified


def get_unified_message(msg: dict, tool_calls_dict: Optional[Dict[str, dict]]) -> dict:
    """Validate and normalize a raw message dict via pydantic + unify."""
    chat_msg = ChatCompletionMessageAdapter.validate_python(msg).model_dump()
    unify_content(chat_msg)
    if chat_msg["role"] in ("system", "assistant", "tool"):
        chat_msg["reasoning_content"] = (
            "" if chat_msg.get("reasoning_content") is None else chat_msg["reasoning_content"]
        )
        chat_msg["tool_calls"] = (
            [] if chat_msg.get("tool_calls") is None else chat_msg["tool_calls"]
        )
        if tool_calls_dict is not None:
            unify_tool_calls(chat_msg, tool_calls_dict)
    return chat_msg


# ========================= Hash Computation =========================


def compute_message_hash(index: int, message: dict, pre_hash: str) -> str:
    """SHA-256 hash of a unified message, chained to previous hash."""
    if message["role"] == "user":
        data = {
            "index": index,
            "role": message["role"],
            "content": message["content"],
            "pre_hash": pre_hash,
        }
    elif message["role"] in ("system", "assistant", "tool"):
        data = {
            "index": index,
            "role": message["role"],
            "content": message["content"],
            "reasoning_content": message.get("reasoning_content", ""),
            "tool_calls": message.get("tool_calls", []),
            "pre_hash": pre_hash,
        }
    else:
        data = {
            "index": index,
            "role": message["role"],
            "content": message["content"],
            "pre_hash": pre_hash,
        }
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# ========================= MessageTracker =========================


class MessageTracker:
    """Per-episode message tracker with incremental tokenization and fork detection."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        enable_thinking: bool = False,
        enable_fork: bool = False,
    ):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.enable_fork = enable_fork

        # Hash-chain data structures
        self.msg_item_dict: Dict[str, MessageItem] = {}
        self.prev_hash_dict: Dict[str, str] = {}
        self.tool_calls_dict: Dict[str, dict] = {}
        self.new_line_token_id: int = tokenizer.encode("\n")[0]

        # Dummy prefix for single-message tokenization
        self.base_chat_history: List[dict] = list(BASE_CHAT_HISTORY)
        self.tools: Optional[List[dict]] = None
        self.base_offset: Optional[int] = None

        # Fork tracking
        self.step_response_hashes: List[str] = []
        self.last_recorded_msg_count: int = 0

        # Cache current pre_hash after process_messages for record_response
        self.current_pre_hash: str = DEFAULT_HASH

        # Per-step cache for step mode
        self.pending_messages: Optional[List[dict]] = None
        # Completed step data: list of {response_ids, logprobs, messages}
        self.step_records: List[dict] = []

    # -------------------- Tools / Base Offset --------------------

    def compute_base_offset(self, tools: Optional[List[dict]]) -> int:
        """Compute the token length of the dummy prefix."""
        ids = self.tokenizer.apply_chat_template(
            self.base_chat_history,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
            enable_thinking=self.enable_thinking,
            return_dict=False
        )
        return len(ids)

    def update_tools(self, tools: List[dict]) -> None:
        """Update tools and recompute base_offset if tools changed."""
        if self.tools != tools:
            self.tools = tools
            self.base_offset = self.compute_base_offset(tools)

    def ensure_base_offset(self) -> None:
        """Lazily compute base_offset on first use."""
        if self.base_offset is None:
            self.base_offset = self.compute_base_offset(self.tools)

    # -------------------- Single Message Tokenization --------------------

    def prepare_message_for_tokenize(self, msg: dict) -> dict:
        """
        Prepare a unified message for tokenization.
        - Restore content from list-of-dicts to str for jinja template compatibility.
        - Convert tool_calls arguments from str to dict for proper jinja template rendering.
        """
        return convert_list_content_str([msg], parse_tool_call_parameter_to_dict=True)[0]

    def tokenize_single_message(
        self,
        msg: dict,
        add_generation_prompt: bool = False,
        tools: Optional[List[dict]] = None,
    ) -> List[int]:
        """Tokenize a single message using dummy prefix + slice."""
        prepared_msg = self.prepare_message_for_tokenize(msg)

        if msg["role"] == "system":
            # System messages: tokenize directly
            return self.tokenizer.apply_chat_template(
                [prepared_msg],
                tokenize=True,
                add_generation_prompt=False,
                tools=tools,
                enable_thinking=self.enable_thinking,
                return_dict=False
            )
        else:
            # user/tool/assistant: dummy prefix + slice
            self.ensure_base_offset()
            full_ids = self.tokenizer.apply_chat_template(
                [*self.base_chat_history, prepared_msg],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                enable_thinking=self.enable_thinking,
                return_dict=False
            )
            return full_ids[self.base_offset:]

    # -------------------- Message Processing --------------------

    def process_messages(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Incrementally process messages, return full input_ids for inference."""
        # 1. Update tools and base_offset
        if tools is not None:
            self.update_tools(tools)

        # 2. Unify messages
        unified_msgs = [
            get_unified_message(m, self.tool_calls_dict) for m in messages
        ]

        # 3. Find already-recorded prefix
        pre_hash = DEFAULT_HASH
        pre_msg_length = 0
        for idx, msg in enumerate(unified_msgs):
            cur_hash = compute_message_hash(idx, msg, pre_hash)
            if cur_hash in self.msg_item_dict:
                pre_msg_length += 1
                pre_hash = cur_hash
            else:
                break

        # 3.5. Fork detection
        if self.last_recorded_msg_count > 0 and pre_msg_length < self.last_recorded_msg_count:
            logger.warning(
                f"History fork detected: matched {pre_msg_length} msgs, "
                f"but last step had {self.last_recorded_msg_count} msgs"
            )

        # 4. Tokenize new messages
        new_msgs = unified_msgs[pre_msg_length:]
        for i, msg in enumerate(new_msgs):
            msg_idx = pre_msg_length + i
            is_last = (msg_idx == len(unified_msgs) - 1)

            if msg["role"] == "assistant":
                logger.warning(
                    f"Assistant message at index {msg_idx} being tokenized "
                    f"(normally should come from record_response)"
                )

            token_ids = self.tokenize_single_message(
                msg,
                add_generation_prompt=(add_generation_prompt and is_last),
                tools=self.tools,
            )

            item = MessageItem(index=msg_idx, message=msg, token_ids=token_ids)
            cur_hash = compute_message_hash(msg_idx, msg, pre_hash)
            self.msg_item_dict[cur_hash] = item
            self.prev_hash_dict[cur_hash] = pre_hash
            pre_hash = cur_hash

        # Cache pre_hash for record_response
        self.current_pre_hash = pre_hash

        # 5. Build full input_ids
        input_ids = self.build_input_ids(unified_msgs)

        # Cache messages for step mode pairing with record_response
        self.pending_messages = messages

        return input_ids

    def build_input_ids(self, unified_msgs: List[dict]) -> List[int]:
        """Concatenate all message token_ids with newline insertion after assistant."""
        input_ids: List[int] = []
        pre_hash = DEFAULT_HASH
        for idx, msg in enumerate(unified_msgs):
            cur_hash = compute_message_hash(idx, msg, pre_hash)
            item = self.msg_item_dict[cur_hash]
            if idx >= 1 and unified_msgs[idx - 1]["role"] == "assistant":
                input_ids.append(self.new_line_token_id)
            input_ids.extend(item.token_ids)
            pre_hash = cur_hash
        return input_ids

    # -------------------- Response Recording --------------------

    def record_response(
        self,
        msg_pos: int,
        resp_msg: dict,
        response_ids: List[int],
        logprobs: Optional[List[float]] = None,
    ) -> None:
        """Record an LLM-generated assistant response with its token_ids from inference."""
        # Record tool_calls for future unification
        if "tool_calls" in resp_msg and isinstance(resp_msg["tool_calls"], list):
            for tc_item in resp_msg["tool_calls"]:
                if "id" in tc_item:
                    self.tool_calls_dict[tc_item["id"]] = tc_item

        unified_msg = get_unified_message(resp_msg, self.tool_calls_dict)
        item = MessageItem(
            index=msg_pos,
            message=unified_msg,
            token_ids=response_ids,
            logprobs=logprobs,
        )

        cur_hash = compute_message_hash(msg_pos, unified_msg, self.current_pre_hash)
        self.msg_item_dict[cur_hash] = item
        self.prev_hash_dict[cur_hash] = self.current_pre_hash

        # Update fork tracking
        self.step_response_hashes.append(cur_hash)
        self.last_recorded_msg_count = msg_pos + 1
        self.current_pre_hash = cur_hash

        # Save step record for step mode extraction
        if self.pending_messages is not None:
            self.step_records.append({
                "response_ids": response_ids,
                "logprobs": logprobs,
                "messages": self.pending_messages,
            })
            self.pending_messages = None

    # -------------------- Trajectory Data Extraction --------------------

    def get_trajectory_data(self, mode: str = "traj") -> List[dict]:
        """
        Extract training data as a list of sample dicts.

        Args:
            mode: "traj" returns aggregated trajectory samples (one per branch);
                  "step" returns one sample per assistant response step.

        Each sample dict contains:
          - token_ids: List[int]
          - response_masks: List[int] (1 for model-generated assistant tokens)
          - logprobs: List[float]
          - messages: List[dict]
        """
        if mode == "step":
            return self._get_step_samples()

        branches = self.identify_branches()
        if not branches:
            return []

        if not self.enable_fork:
            # Use the longest branch only
            longest = max(branches, key=len)
            branches = [longest]

        results = []
        for branch in branches:
            leaf_hash = branch[-1]
            path = self.trace_path_to_root(leaf_hash)
            sample = self.build_sample_from_path(path)
            results.append(sample)
        return results

    def _get_step_samples(self) -> List[dict]:
        """Extract per-step training samples from cached step records.

        For each step, re-tokenizes the history messages via apply_chat_template
        to get prompt_ids, and uses response_ids from the inference engine.
        """
        results: List[dict] = []
        for record in self.step_records:
            messages = record.get("messages", [])
            response_ids = record["response_ids"]
            resp_logprobs = record["logprobs"]

            # Re-tokenize history messages as a whole prompt
            prompt_ids = self.tokenizer.apply_chat_template(
                convert_list_content_str(messages, parse_tool_call_parameter_to_dict=True),
                tools=self.tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                return_dict=False
            )

            token_ids = prompt_ids + response_ids
            response_masks = [0] * len(prompt_ids) + [1] * len(response_ids)
            logprobs = [0.0] * len(prompt_ids) + (resp_logprobs if resp_logprobs else [0.0] * len(response_ids))

            results.append({
                "token_ids": token_ids,
                "response_masks": response_masks,
                "logprobs": logprobs,
                "messages": messages,
            })
        return results

    def identify_branches(self) -> List[List[str]]:
        """
        Group step_response_hashes into branches.
        Consecutive steps belong to the same branch iff the previous step's
        response_hash is an ancestor of the current step's hash chain.
        """
        if not self.step_response_hashes:
            return []

        branches: List[List[str]] = [[self.step_response_hashes[0]]]

        for i in range(1, len(self.step_response_hashes)):
            prev_hash = self.step_response_hashes[i - 1]
            curr_hash = self.step_response_hashes[i]

            if self.is_ancestor(prev_hash, curr_hash):
                branches[-1].append(curr_hash)
            else:
                branches.append([curr_hash])

        return branches

    def is_ancestor(self, ancestor_hash: str, descendant_hash: str) -> bool:
        """Check if ancestor_hash is on the path from descendant_hash back to root."""
        h = descendant_hash
        while h != DEFAULT_HASH:
            if h == ancestor_hash:
                return True
            h = self.prev_hash_dict.get(h, DEFAULT_HASH)
        return False

    def trace_path_to_root(self, leaf_hash: str) -> List[str]:
        """Walk from leaf to root, return ordered hash list (root -> leaf)."""
        path: List[str] = []
        h = leaf_hash
        while h != DEFAULT_HASH:
            path.append(h)
            h = self.prev_hash_dict[h]
        path.reverse()
        return path

    def build_sample_from_path(self, path: List[str]) -> dict:
        """Build a training sample from an ordered hash path."""
        token_ids: List[int] = []
        response_masks: List[int] = []
        logprobs: List[float] = []
        messages: List[dict] = []

        prev_role: Optional[str] = None
        for hash_key in path:
            item = self.msg_item_dict[hash_key]

            # Insert newline after assistant
            if prev_role == "assistant":
                token_ids.append(self.new_line_token_id)
                response_masks.append(0)
                logprobs.append(0.0)

            token_ids.extend(item.token_ids)

            # response_mask: 1 only for model-generated assistant (has logprobs)
            if item.role == "assistant" and item.logprobs is not None:
                response_masks.extend([1] * len(item.token_ids))
            else:
                response_masks.extend([0] * len(item.token_ids))

            if item.logprobs is not None:
                logprobs.extend(item.logprobs)
            else:
                logprobs.extend([0.0] * len(item.token_ids))

            # Collect messages
            messages.append(item.message)

            prev_role = item.role

        return {
            "token_ids": token_ids,
            "response_masks": response_masks,
            "logprobs": logprobs,
            "messages": messages,
        }
