import sys
from types import SimpleNamespace
from types import SimpleNamespace as _SimpleNamespace

import numpy as np
import pytest
import torch

from roll.datasets.collator import DataCollatorForDiffusion
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.utils import get_diffusion_encode_function

QWEN_IMAGE_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)


class _FakeTokenizer:
    """Fake tokenizer that mimics HuggingFace tokenizer behavior.

    - apply_chat_template(tokenize=False) → returns concatenated text string
    - __call__(text_list) → returns dict with input_ids and attention_mask
    """
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        # When tokenize=False, return a text string (like real tokenizers do)
        text = "".join(message["content"] for message in messages)
        return text

    def __call__(self, text_list, **kwargs):
        # Batch tokenize: each text → list of token IDs
        input_ids = []
        attention_mask = []
        for text in text_list:
            ids = [ord(c) % 100 for c in text]  # deterministic pseudo-tokenization
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_diffusion_encode_function_builds_positive_and_negative_prompts():
    tokenizer = _FakeTokenizer()
    data_args = SimpleNamespace(
        messages="prompt",
        system_prompt=QWEN_IMAGE_SYSTEM_PROMPT,
        template="native",
    )
    encode = get_diffusion_encode_function(
        template_name="native",
        tokenizer=tokenizer,
        data_args=data_args,
    )

    encoded = encode({"system_prompt": [QWEN_IMAGE_SYSTEM_PROMPT], "prompt": ["draw a cat"], "negative_prompt": [""]})

    assert encoded["input_ids"] == encoded["prompt_ids"]
    assert len(encoded["prompt_ids"]) == 1
    assert len(encoded["negative_prompt_ids"]) == 1
    assert "encode_start_idx" in encoded
    assert encoded["encode_start_idx"][0] > 0
    # Positive and negative should differ (different content → different tokens)
    assert encoded["prompt_ids"][0] != encoded["negative_prompt_ids"][0]


def test_diffusion_encode_function_without_system_prompt():
    tokenizer = _FakeTokenizer()
    data_args = SimpleNamespace(
        prompt="prompt",
        system_prompt="",
        template="native",
    )
    encode = get_diffusion_encode_function(
        template_name="native",
        tokenizer=tokenizer,
        data_args=data_args,
    )

    encoded = encode({"system_prompt": [""], "prompt": ["draw a cat"], "negative_prompt": [""]})

    assert len(encoded["prompt_ids"]) == 1
    assert len(encoded["negative_prompt_ids"]) == 1
    # Without system_prompt, encode_start_idx should be 0
    assert encoded["encode_start_idx"][0] == 0
    # Without system_prompt, raw text is tokenized directly
    assert encoded["prompt_ids"][0] == [ord(c) % 100 for c in "draw a cat"]


def test_data_collator_for_diffusion_outputs_multi_modal_data():
    collator = DataCollatorForDiffusion(tokenizer=_FakeTokenizer(), max_length=6)
    batch = collator(
        [
            {
                "input_ids": [11, 12],
                "attention_mask": [1, 1],
                "prompt_ids": [21, 22],
                "prompt_mask": [1, 1],
                "negative_prompt_ids": [31, 32],
                "negative_prompt_mask": [1, 1],
                "domain": "ocr",
                "id": "split-0",
                "ground_truth": "HELLO",
                "encode_start_idx": 34,
            }
        ]
    )

    assert torch.equal(batch["input_ids"], torch.tensor([[11, 12, 0, 0, 0, 0]], dtype=torch.long))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.long))
    assert batch["domain"].tolist() == ["ocr"]
    assert batch["id"].tolist() == ["split-0"]
    assert batch["ground_truth"].tolist() == ["HELLO"]
    assert batch["multi_modal_data"].shape == (1,)
    assert torch.equal(batch["multi_modal_data"][0]["prompt_ids"], torch.tensor([21, 22], dtype=torch.long))
    assert torch.equal(batch["multi_modal_data"][0]["prompt_mask"], torch.tensor([1, 1], dtype=torch.long))
    assert torch.equal(batch["multi_modal_data"][0]["negative_prompt_ids"], torch.tensor([31, 32], dtype=torch.long))
    assert torch.equal(batch["multi_modal_data"][0]["negative_prompt_mask"], torch.tensor([1, 1], dtype=torch.long))
    assert batch["multi_modal_data"][0]["encode_start_idx"] == 34


def test_router_client_normalizes_legacy_vllm_omni_multi_modal_data_lists():
    if "more_itertools" not in sys.modules:
        sys.modules["more_itertools"] = _SimpleNamespace(chunked=lambda seq, n: [seq[i:i + n] for i in range(0, len(seq), n)])
    try:
        from roll.distributed.scheduler.router import RouterClient
    except ModuleNotFoundError as exc:
        pytest.skip(f"router import requires optional runtime dependency: {exc.name}")

    mm_data = np.empty(1, dtype=object)
    mm_data[0] = {
        "prompt_ids": [21, 22],
        "prompt_mask": [1, 1],
        "negative_prompt_ids": [31, 32],
        "negative_prompt_mask": [1, 1],
        "encode_start_idx": 34,
    }
    req = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        non_tensors={"multi_modal_data": mm_data},
        meta_info={"generation_config": {"num_return_sequences": 1, "max_new_tokens": 1}},
    )
    client = RouterClient(
        proxy=None,
        meta={"strategy_name": "vllm_omni", "eos_token_id": 151645, "pad_token_id": 0},
    )

    payload, _ = client._preprocess_generate(req, request_id="rid0")

    assert torch.equal(payload["multi_modal_data"]["prompt_ids"], torch.tensor([21, 22], dtype=torch.long))
    assert torch.equal(payload["multi_modal_data"]["prompt_mask"], torch.tensor([1, 1], dtype=torch.long))
    assert torch.equal(payload["multi_modal_data"]["negative_prompt_ids"], torch.tensor([31, 32], dtype=torch.long))
    assert torch.equal(payload["multi_modal_data"]["negative_prompt_mask"], torch.tensor([1, 1], dtype=torch.long))
    assert payload["multi_modal_data"]["encode_start_idx"] == 34


def test_vllm_omni_postprocess_keeps_tensor_outputs_out_of_meta_info():
    from roll.pipeline.diffusion.rollout_loop import DiffusionRolloutLoop

    loop = DiffusionRolloutLoop()
    request = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[11, 12, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        },
        meta_info={
            "generation_config": {"num_return_sequences": 1},
            "global_step": 0,
        },
    )
    data = DataProto(
        meta_info={
            "responses": torch.rand(1, 3, 4, 4),
            "rollout_log_probs": torch.rand(1, 2),
            "all_timesteps": torch.rand(1, 2),
            "all_latents": torch.rand(1, 3, 8),
            "prompt_embeds": torch.rand(1, 2, 16),
            "prompt_embeds_mask": torch.ones(1, 2),
            "negative_prompt_embeds": torch.rand(1, 2, 16),
            "negative_prompt_embeds_mask": torch.ones(1, 2),
            "eos_token_id": [151645, 0],
            "pad_token_id": 0,
        }
    )

    output = loop.postprocess_output_data(request, data, sequence_length=3)

    assert "prompt_embeds" in output.batch.keys()
    assert "negative_prompt_embeds" in output.batch.keys()
    assert "prompt_embeds" not in output.meta_info
    assert "negative_prompt_embeds" not in output.meta_info
    assert output.meta_info["pad_token_id"] == 0
