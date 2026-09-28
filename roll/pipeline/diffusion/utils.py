"""Common diffusion training utilities shared across algorithms."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple

import numpy as np
import torch

from roll.datasets.chat_template import get_chat_template
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.schema import require_batch_keys, require_non_tensor_keys
from roll.utils.logging import get_logger

if TYPE_CHECKING:
    from roll.pipeline.diffusion.diffusion_config import DiffusionConfig

logger = get_logger()


# ========================= Group / Advantage =========================


def attach_group_ids(batch: DataProto) -> DataProto:
    """Assign a sequential group_id to each unique sample id."""
    sample_ids = batch.non_tensor_batch.get("id")
    if sample_ids is None:
        raise ValueError("Rollout batch missing non_tensor_batch['id']; cannot build group_id.")
    if hasattr(sample_ids, "tolist"):
        sample_ids = sample_ids.tolist()

    group_id_map: dict[object, int] = {}
    group_ids: list[int] = []
    next_group_id = 0
    for sample_id in sample_ids:
        if sample_id not in group_id_map:
            group_id_map[sample_id] = next_group_id
            next_group_id += 1
        group_ids.append(group_id_map[sample_id])

    group_ids_array = np.empty(len(group_ids), dtype=object)
    group_ids_array[:] = group_ids
    batch.non_tensor_batch["group_id"] = group_ids_array
    return batch


def resolve_group_ids(batch: DataProto) -> List[int]:
    """Read group_id from non_tensor_batch and return as list[int]."""
    if "group_id" not in batch.non_tensor_batch:
        raise KeyError("Batch requires non_tensor_batch['group_id'].")
    values = batch.non_tensor_batch["group_id"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


def compute_per_tag_score_metrics(batch: DataProto, prefix: str = "reward") -> Dict[str, float]:
    """Aggregate ``scores`` per task tag into {prefix}/{tag}/score/{mean,max,min}.

    Grouping is by ``tag``, not ``domain``: ``domain`` only says which reward
    worker scores the sample and several tags may share one worker (all geneval
    subtasks do), so ``tag`` is what carries the task identity. It is supplied by
    DataCollatorForDiffusion.

    Also emits an ``overall`` group over the whole batch. Returns an empty dict
    when the batch carries no scores. Computed in the pipeline rather than in a
    reward worker so metrics never travel through ``meta_info``, which the
    generate scheduler would blindly re-aggregate into ``.../mean/mean`` keys.
    """
    if "scores" not in batch.batch.keys():
        return {}
    if "tag" not in batch.non_tensor_batch:
        raise KeyError("Batch missing non_tensor_batch['tag']; DataCollatorForDiffusion must provide it.")
    raw = batch.batch["scores"].float()
    scores = raw.reshape(raw.shape[0], -1).mean(dim=1).tolist()
    tags = batch.non_tensor_batch["tag"]
    tags = tags.tolist() if hasattr(tags, "tolist") else tags
    metrics: Dict[str, float] = {
        f"{prefix}/overall/score/mean": sum(scores) / len(scores),
        f"{prefix}/overall/score/max": max(scores),
        f"{prefix}/overall/score/min": min(scores),
    }
    tag_to_scores: Dict[str, List[float]] = {}
    for tag, score in zip(tags, scores):
        tag_to_scores.setdefault(str(tag), []).append(score)
    for tag, values in sorted(tag_to_scores.items()):
        metrics[f"{prefix}/{tag}/score/mean"] = sum(values) / len(values)
        metrics[f"{prefix}/{tag}/score/max"] = max(values)
        metrics[f"{prefix}/{tag}/score/min"] = min(values)
    return metrics


def _as_advantage_scores(scores: torch.Tensor, num_steps: int | None) -> torch.Tensor:
    if scores.ndim == 1:
        if num_steps is None:
            raise ValueError("num_steps is required when scores is [B].")
        return scores.float().unsqueeze(-1).expand(scores.shape[0], num_steps).clone()

    if scores.ndim != 2:
        raise ValueError(f"scores must be [B], [B,1] or [B,T], got {tuple(scores.shape)}")

    scores = scores.float()
    if scores.shape[1] == 1:
        if num_steps is None:
            return scores.clone()
        return scores.expand(scores.shape[0], num_steps).clone()

    if num_steps is not None and scores.shape[1] != num_steps:
        raise ValueError(f"scores.shape[1]={scores.shape[1]} must equal num_steps={num_steps}")

    return scores.clone()


def compute_grpo_outcome_advantage(
    *,
    scores: torch.Tensor,
    group_ids: Iterable[int],
    num_steps: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-normalized advantage computation (standard GRPO)."""
    advantages = _as_advantage_scores(scores, num_steps).to(dtype=torch.float32)
    batch_size = advantages.shape[0]

    group_ids_list = [int(g) for g in group_ids]
    if len(group_ids_list) != batch_size:
        logger.error(
            "grpo/adv_error: group_ids length mismatch, len(group_ids)=%s batch_size=%s",
            len(group_ids_list),
            batch_size,
        )
        raise ValueError(f"len(group_ids)={len(group_ids_list)} must equal batch size={batch_size}")

    unique_group_ids = sorted(set(group_ids_list))

    epsilon = 1e-4
    group_means = {}
    group_stds = {}

    for group_id in unique_group_ids:
        idx = [i for i, gid in enumerate(group_ids_list) if gid == group_id]
        if len(idx) == 0:
            logger.error("grpo/adv_error: empty group encountered for group_id=%s", group_id)
            raise ValueError(f"Empty group encountered for group_id={group_id}")
        group_scores = advantages[idx]
        group_mean = group_scores.mean()
        # torch.std defaults to unbiased=True, which returns NaN for n=1 (division by n-1=0).
        group_std = group_scores.std() if len(idx) > 1 else torch.tensor(0.0, dtype=advantages.dtype, device=advantages.device)
        group_means[group_id] = group_mean
        group_stds[group_id] = group_std

    for i, group_id in enumerate(group_ids_list):
        advantages[i] = (advantages[i] - group_means[group_id]) / (group_stds[group_id] + epsilon)

    returns = advantages.clone()
    return advantages, returns


def compute_group_advantage_metrics(batch: DataProto) -> Dict[str, float]:
    """Compute per-group advantage mean/std metrics."""
    if "advantages" not in batch.batch or not torch.is_tensor(batch.batch["advantages"]):
        return {}

    advantages = batch.batch["advantages"]
    sample_advantages = advantages.float() if advantages.ndim == 1 else advantages.float().reshape(advantages.shape[0], -1).mean(dim=1)
    group_ids = resolve_group_ids(batch)
    if len(group_ids) != sample_advantages.shape[0]:
        raise ValueError(
            "Group advantage metrics require one group_id per sample, "
            f"got len(group_ids)={len(group_ids)} num_samples={sample_advantages.shape[0]}"
        )

    group_sums: Dict[int, float] = {}
    group_sq_sums: Dict[int, float] = {}
    group_counts: Dict[int, int] = {}
    for idx, gid in enumerate(group_ids):
        adv_val = float(sample_advantages[idx].item())
        group_sums[gid] = group_sums.get(gid, 0.0) + adv_val
        group_sq_sums[gid] = group_sq_sums.get(gid, 0.0) + adv_val * adv_val
        group_counts[gid] = group_counts.get(gid, 0) + 1

    metrics: Dict[str, float] = {}
    for gid in sorted(group_sums.keys()):
        count = group_counts[gid]
        mean = group_sums[gid] / count
        variance = max(0.0, group_sq_sums[gid] / count - mean * mean)
        metrics[f"grpo_group/advantage/group_{gid}/mean"] = mean
        metrics[f"grpo_group/advantage/group_{gid}/std"] = variance ** 0.5
    return metrics


# ========================= Training Batch Preparation =========================


def attach_default_log_probs(batch: DataProto) -> DataProto:
    """Reuse rollout_log_probs as old_log_probs and ref_log_probs."""
    logger.info("DiffusionPipeline reusing rollout_log_probs as old_log_probs")
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
    batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
    return batch


def prepare_training_batch(
    *,
    batch: DataProto,
    pipeline: "DiffusionPipeline",
    global_step: int,
) -> tuple[DataProto, dict[str, float]]:
    """Prepare a training batch from rollout output; dispatch by algorithm."""
    algorithm = pipeline.pipeline_config.algorithm

    batch.meta_info["global_step"] = global_step
    batch.meta_info["is_offload_states"] = pipeline.pipeline_config.is_offload_states
    batch.meta_info["is_offload_optimizer_states_in_train_step"] = pipeline.pipeline_config.is_offload_optimizer_states_in_train_step

    if algorithm == "flowgrpo":
        return _prepare_flowgrpo_training_batch(batch=batch, pipeline=pipeline, global_step=global_step)
    if algorithm in {"diffnft"}:
        return _prepare_diffnft_training_batch(batch=batch, pipeline=pipeline, global_step=global_step)
    raise ValueError(f"Unsupported algorithm={algorithm!r}")


def _prepare_flowgrpo_training_batch(
    *,
    batch: DataProto,
    pipeline: "DiffusionPipeline",
    global_step: int,
) -> tuple[DataProto, dict[str, float]]:
    """FlowGRPO: group-normalized advantages over the full SDE trajectory."""
    context = "Training batch preparation"
    require_batch_keys(batch, ["token_level_rewards"], context)
    require_non_tensor_keys(batch, ["group_id"], context)

    num_steps = int(batch.batch["all_timesteps"].shape[1])
    batch.batch["flow_loss_mask"] = torch.ones_like(batch.batch["rollout_log_probs"], dtype=torch.long)
    batch.meta_info["loss_mask_keys"] = ["flow_loss_mask"]

    advantages, returns = compute_grpo_outcome_advantage(
        scores=batch.batch["scores"],
        group_ids=resolve_group_ids(batch),
        num_steps=num_steps,
    )
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = returns

    batch = attach_default_log_probs(batch)
    return batch, compute_group_advantage_metrics(batch)


def _prepare_diffnft_training_batch(
    *,
    batch: DataProto,
    pipeline: "DiffusionPipeline",
    global_step: int,
) -> tuple[DataProto, dict[str, float]]:
    """DiffNFT: scores -> group normalization -> optimality probability r in [0, 1].

    Paper Algorithm 1: rnorm = rraw - group_mean (the same group normalization
    as GRPO), then r = 0.5 + 0.5 * clip(rnorm / Zc, -1, 1).
    ``compute_grpo_outcome_advantage`` already normalizes by the batch std, so
    Zc = batch std here and the clip is applied directly. The result is stored
    in the ``advantages`` field and read as ``r`` by ActorNFTWorker. DiffNFT
    does not replay rollout log-probs, so no log-prob fields are attached.
    """
    context = "DiffNFT training batch preparation"
    require_batch_keys(batch, ["scores"], context)
    require_non_tensor_keys(batch, ["group_id"], context)

    # DiffNFT has no loss mask; the strategy requires the key to be present.
    batch.meta_info["loss_mask_keys"] = []

    advantages, _ = compute_grpo_outcome_advantage(
        scores=batch.batch["scores"],
        group_ids=resolve_group_ids(batch),
        num_steps=1,
    )
    # Map normalized advantages to optimality probability r in [0, 1] using
    # adv_clip_max as the saturation bound.  With adv_clip_max=5.0, GRPO
    # advantages (mean 0, std ~1) map to r in ~[0.4, 0.6], providing smooth
    # gradient signals.  A hard [-1, 1] clip would saturate almost all
    # samples in a 16-member group and collapse the loss to binary.
    adv_clip_max = float(pipeline.pipeline_config.adv_clip_max)
    clipped = advantages.float().clamp(min=-adv_clip_max, max=adv_clip_max)
    batch.batch["advantages"] = (clipped / adv_clip_max) / 2.0 + 0.5

    return batch, compute_group_advantage_metrics(batch)


# ========================= Config Validation =========================


def validate_diffusion_config(pipeline_config: "DiffusionConfig") -> None:
    """Validate diffusion pipeline configuration."""
    if pipeline_config.num_return_sequences_in_group <= 1:
        raise ValueError("num_return_sequences_in_group must be greater than 1.")
    if not pipeline_config.enable_reference and pipeline_config.use_kl_loss:
        logger.warning(
            "Config warning: enable_reference=False while use_kl_loss=True. "
            "old_log_probs will be copied into ref_log_probs, so KL is expected to be near zero."
        )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def validate_diffusion_dataset(dataset) -> None:
    """Validate that the preprocessed dataset has all required columns.

    Raises ValueError if any required column is missing.
    """
    required = [
        "domain",
        "id",
        "ground_truth",
        "prompt_ids",
        "prompt_mask",
        "negative_prompt_ids",
        "negative_prompt_mask",
        "encode_start_idx",
    ]
    missing = [key for key in required if key not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Diffusion dataset missing required columns after preprocessing: "
            f"missing={missing} available={dataset.column_names}"
        )


# ========================= Dataset Encoding =========================


def compute_encode_start_idx(template_func, tokenizer, system_prompt: str) -> int:
    """Compute the number of prefix tokens (system message + user header) before user content.

    Tokenizes two conversations that differ only in user content ("A" vs "B"),
    then returns the length of their common token prefix. This gives the exact
    number of tokens corresponding to the chat template structure before the
    user's actual content begins.
    """
    text_a = template_func([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "A"},
    ])
    text_b = template_func([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "B"},
    ])
    ids_a = tokenizer([text_a])["input_ids"][0]
    ids_b = tokenizer([text_b])["input_ids"][0]

    min_len = min(len(ids_a), len(ids_b))
    for i in range(min_len):
        if ids_a[i] != ids_b[i]:
            return i
    return min_len


def get_diffusion_encode_function(template_name, tokenizer, data_args=None, tag_to_template=None):
    """Build a batched encode function for diffusion rollout datasets.

    Expects the pre-processed dataset format where each row has:
    - ``system_prompt``: str — system message content
    - ``prompt``: str — user message content (positive prompt)
    - ``negative_prompt``: str — user message content (negative prompt)

    The function constructs [system, user] conversations, applies the chat
    template, and tokenizes. Both positive and negative prompts share the
    same system_prompt from the dataset row.

    Args:
        template_name: Chat template key (e.g. "native", "qwen2_5").
        tokenizer: Tokenizer instance.
        data_args: Unused, kept for signature compatibility.
        tag_to_template: Unused, kept for signature compatibility.

    Returns:
        A callable suitable for ``datasets.Dataset.map(batched=True)``.
    """
    template_func = get_chat_template(template_name, tokenizer)

    def _encode_texts(system_prompts, user_texts):
        """Construct [system, user] conversations and apply chat template."""
        text_list = []
        for sp, ut in zip(system_prompts, user_texts):
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": ut},
            ]
            text_list.append(template_func(messages))
        return text_list

    def encode_function(data_i):
        system_prompts = data_i["system_prompt"]
        prompts = data_i["prompt"]
        negative_prompts = data_i["negative_prompt"]

        # Compute encode_start_idx from the first system_prompt in this batch.
        first_sp = system_prompts[0] if system_prompts else ""
        encode_start_idx = compute_encode_start_idx(template_func, tokenizer, first_sp)
        logger.info(
            "compute_encode_start_idx: system_prompt_len=%d encode_start_idx=%d",
            len(first_sp), encode_start_idx,
        )

        prompt_text_list = _encode_texts(system_prompts, prompts)
        prompt_encodings = tokenizer(prompt_text_list) if prompt_text_list else {"input_ids": [], "attention_mask": []}

        negative_text_list = _encode_texts(system_prompts, negative_prompts)
        negative_encodings = tokenizer(negative_text_list) if negative_text_list else {"input_ids": [], "attention_mask": []}

        return {
            "input_ids": prompt_encodings["input_ids"],
            "attention_mask": prompt_encodings["attention_mask"],
            "prompt_ids": prompt_encodings["input_ids"],
            "prompt_mask": prompt_encodings["attention_mask"],
            "negative_prompt_ids": negative_encodings["input_ids"],
            "negative_prompt_mask": negative_encodings["attention_mask"],
            "encode_start_idx": [encode_start_idx] * len(system_prompts),
        }

    return encode_function
