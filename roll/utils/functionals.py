from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roll.distributed.scheduler.protocol import DataProto
import enum
import traceback
import heapq
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from roll.configs.base_config import PPOConfig
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.platforms import current_platform
from roll.utils.kl_controller import AdaptiveKLController
from roll.utils.logging import get_logger

logger = get_logger()


def tensor_to_cpu_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cpu:
            obj.data = obj.data.detach().cpu()
        return True
    return False


def tensor_to_cuda_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cuda:
            obj.data = obj.data.detach().to(device=torch.device(current_platform.device_type))
        return True
    return False


def delete_tensor_grad_visitor(obj, path):
    if torch.is_tensor(obj):
        obj.grad = None
        return True
    return False


def traverse_obj(value, visitor, path=()):
    """
    遍历对象的所有属性，包括属性的属性，找到所有的 Tensor。
    :param value: 任意 Python 对象
    :visitor
    :path
    """
    if visitor(value, path):
        return
    elif isinstance(value, dict):
        for key, value in value.items():
            traverse_obj(value, visitor, path + (str(key),))
    elif isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            traverse_obj(item, visitor, path + (index,))
    elif hasattr(value, "__dict__"):
        for attr_name in dir(value):
            if not attr_name.startswith("__"):
                try:
                    attr_value = getattr(value, attr_name)
                    traverse_obj(attr_value, visitor, path + (f"attr:{attr_name}",))
                except Exception as e:
                    logger.error(e)
                    continue


def union_two_dict(dict1: Dict, dict2: Dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def divide_by_chunk_size(
    data: Union[np.ndarray, TensorDict], chunk_sizes: List[int]
) -> List[Union[np.ndarray, TensorDict]]:
    """
    将numpy数组按照chunks的大小切分
    """
    if not isinstance(data, (np.ndarray, TensorDict)):
        raise TypeError("Input 'array' must be a numpy ndarray or a TensorDict.")

    if not all(isinstance(size, int) and size > 0 for size in chunk_sizes):
        raise ValueError("All chunk sizes must be positive integers.")

    total_size = sum(chunk_sizes)
    if total_size != len(data):
        raise ValueError(f"The sum of chunk_sizes ({total_size}) does not match the size of the array ({len(data)}).")

    split_data = []
    start_index = 0
    for size in chunk_sizes:
        end_index = start_index + size
        split_data.append(data[start_index:end_index])
        start_index = end_index
    return split_data


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


def compute_clip_fraction(values: torch.Tensor, clip_max: float, clip_min: float):
    numel = values.numel()
    num_clipped = (values > clip_max).sum().item() + (values < clip_min).sum().item()
    clipfrac = num_clipped / numel if numel > 0 else 0.0
    return clipfrac


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_penalty: str = "kl",
) -> torch.Tensor:
    """
    ref: https://github.com/OpenRLHF/OpenRLHF/blob/494850f50342ed38d5ae76ef45a3207f3523b582/openrlhf/models/utils.py#L7
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    """
    if kl_penalty == "kl":
        log_ratio = log_probs - log_probs_base
    elif kl_penalty == "abs":
        log_ratio = (log_probs - log_probs_base).abs()
    elif kl_penalty == "mse":
        log_ratio = 0.5 * (log_probs - log_probs_base).square()
    elif kl_penalty == "k3":
        kl = log_probs_base - log_probs
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        log_ratio = torch.clamp(kld, min=-10, max=10)
    elif kl_penalty == "full":
        log_ratio = F.kl_div(log_probs_base, log_probs, log_target=True, reduction="none").sum(-1)
    else:
        raise NotImplementedError

    if action_mask is not None:
        return log_ratio * action_mask

    return log_ratio


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    logits = logits.float()
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str, batch_num_tokens: int = None,
             global_valid_samples: int = None, weights: Optional[torch.Tensor] = None):
    """
    ref: https://github.com/volcengine/verl/blob/78532923368aeb058f62201489546d013df47710/verl/trainer/ppo/core_algos.py#L370
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "seq-mean-token-sum" is the default behavior
        weights: `torch.Tensor`
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if batch_num_tokens is None:
        batch_num_tokens = loss_mask.sum()
    if global_valid_samples is None:
        global_valid_samples = loss_mat.size(0)
    if loss_agg_mode == "token-mean":
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (loss_mat * weights.unsqueeze(-1)).sum() / batch_num_tokens
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = masked_sum(loss_mat, loss_mask, dim=-1)  # token-sum
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (global_valid_samples + 1e-8)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1)
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (global_valid_samples + 1e-8)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = masked_sum(loss_mat, loss_mask, dim=-1)
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        mask_sum = mask.sum(axis=dim)
        return torch.where(mask_sum > 0, (tensor * mask).sum(axis=dim) / (mask_sum + 1e-8), torch.zeros_like(mask_sum))
    else:
        return (tensor * mask).sum() / (mask.sum() + 1e-8)

def masked_sum(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        mask_sum = mask.sum(axis=dim)
        return torch.where(mask_sum > 0, (tensor * mask).sum(axis=dim), torch.zeros_like(mask_sum))
    else:
        return (tensor * mask).sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    """
    e.g. end of sentence token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def get_pad_mask(response_id: torch.Tensor, pad_token: int = 0, eos_token: int = 1, dtype=torch.int64):
    """
    e.g. pad token=0
    response_id: [1, 2, 2, 42, 3, 5, 1, 0, 0]
    pad_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]

    If eos_token == pad_token, the first pad token (which is the eos token) should be kept.
    e.g. pad_token=0, eos_token=0
    response_id: [1, 2, 2, 42, 3, 5, 0, 0, 0]
    pad_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]  (first pad token/eos token is kept)
    """
    pad_mask = response_id.not_equal(pad_token).to(dtype)

    # eos_token == pad_token, 需要保留第一个pad token否则会误将eos token mask掉
    if eos_token == pad_token:
        pad_positions = response_id.eq(pad_token).to(dtype)
        cumsum_pad = torch.cumsum(pad_positions, dim=-1)
        first_pad_token = (cumsum_pad == 1).to(dtype)
        pad_mask = pad_mask | first_pad_token

    assert (
        not (pad_mask[:, 0] == 0).logical_and(pad_mask.sum(-1) != 0).any()
    ), f"response_id is not valid: {response_id}, pad_token is {pad_token}"
    return pad_mask


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim).unsqueeze(-1)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim).unsqueeze(-1)
    return mean_centered * var.clamp(min=eps).rsqrt()


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def response_level_masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    # 考虑response的影响？
    mean = masked_mean(values, mask, dim=-1)
    var = masked_var(mean, mask)
    mean = mean.mean()
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def reduce_metrics(metrics: dict, reduce_func=np.mean) -> dict:
    """
    Reduce metrics by parsing an aggregation instruction from the metric name.

    Aggregation can be specified in the metric name using either of the following formats:
      - Suffix after '@': e.g., "loss@sum", "latency@p99"
      - Underscore suffix: e.g., "loss_sum", "latency_p99"

    Supported aggregation tags/suffixes: mean, max, min, p50, p99, std, sum

    Notes:
      - The original metric key is preserved (the '@tag' or '_suffix' remains in the key).
      - Scalar values (int, float, np.number) and torch.Tensor objects are left unchanged.
      - Values of type list, tuple, or np.ndarray are reduced using the inferred aggregation function.
      - If no aggregation tag or suffix is found, the default `reduce_func` is used.
      - Empty sequences are skipped and not modified.
    """
    import numpy as np

    reducers = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "p50": lambda x: np.percentile(x, 50),
        "p99": lambda x: np.percentile(x, 99),
        "std": np.std,
        "sum": np.sum,
    }

    def _parse_aggregation_func(metric_name: str):
        # First, check for '@' separator
        if "@" in metric_name:
            _, tag = metric_name.rsplit("@", 1)
            tag = tag.strip()
            if tag in reducers:
                return reducers[tag]
            else:
                raise ValueError(f"Unknown reducer tag '{tag}' in metric '{metric_name}'")

        # Otherwise, check for underscore-based suffixes
        for suffix_key in ["mean", "max", "min", "p50", "p99", "std", "sum"]:
            if metric_name.endswith(f"_{suffix_key}"):
                return reducers[suffix_key]

        # No aggregation specifier found → use default
        return reduce_func

    for key, val in list(metrics.items()):
        # Skip reduction for scalars and tensors
        if isinstance(val, (int, float, np.number)) or isinstance(val, torch.Tensor):
            continue

        # Reduce sequences
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                continue
            agg_func = _parse_aggregation_func(key)
            metrics[key] = float(agg_func(val))
        else:
            # Fallback for other types (e.g., single-element containers)
            metrics[key] = float(reduce_func(val))

    return metrics

def reduce_metrics_list(metrics_list: list, reduce_func=np.mean) -> dict:
    if len(metrics_list) == 0:
        return {}
    merged_metrics = {k: reduce_func([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    return merged_metrics


def pad_to_length(tensor: torch.Tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(0, length)
        return tensor[indices]
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
        )


def concatenate_input_and_output(input_ids, output_ids, num_return_sequences):
    batch_size, input_seq_len = input_ids.size()
    _, output_seq_len = output_ids.size()
    repeated_input_ids = (
        input_ids.unsqueeze(1)
        .repeat(1, num_return_sequences, 1)
        .view(batch_size * num_return_sequences, input_seq_len)
    )
    sequences = torch.cat((repeated_input_ids, output_ids), dim=1)
    return sequences


def compute_reinforce_return(token_level_rewards: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor):
    with torch.no_grad():
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        cumulative_reward = 0
        for t in reversed(range(gen_len)):
            local_reward = token_level_rewards[:, t] if t < gen_len else 0.0
            cumulative_reward = local_reward + gamma * cumulative_reward
            advantages_reversed.append(cumulative_reward)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages
    return advantages, returns


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor, values: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        gamma: `(float)`
            discounted factor used in RL
        lambd: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values

    return advantages, returns


def expand_to_token_level(data: "DataProto"):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    batch_size = data.batch.batch_size[0]
    # expand as token_level_rewards
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]
    if position_ids.dim() == 3:
        # qwen2vl, (bsz, 3, seqlen), 0/1/2 is same for text, while values of
        # position_ids for text cannot stand for index of tokens, thus use the
        # right padding attention_mask to calculate eos index or `argmax` rather
        # than `max` of position_ids to calculate eos index
        position_ids = position_ids[:, 0]
    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
    token_level_rewards = torch.zeros_like(attention_mask, dtype=response_level_rewards.dtype)  # (bsz, seqlen)

    token_level_rewards[torch.arange(batch_size), eos_mask_idx] = response_level_rewards

    # select the response part
    token_level_rewards = token_level_rewards[:, 1:]

    return token_level_rewards


def reward_norm(
    response_level_rewards: torch.Tensor, n_sample=-1, running_ctrl={}, norm_mean_type=None, norm_std_type=None
):
    group_mode = (norm_mean_type == "group") or (norm_std_type == "group")
    if group_mode and n_sample > 0:
        reshape_reward = response_level_rewards.reshape(*response_level_rewards.size()[:-1], -1, n_sample)
    if norm_mean_type == "running" or norm_std_type == "running":
        running = running_ctrl["domain"]
        running.update(response_level_rewards)
    # 均值计算
    if norm_mean_type == "batch":
        reward_mean = response_level_rewards.mean()
    elif norm_mean_type == "group":
        reward_mean = reshape_reward.mean(dim=-1, keepdim=True)
    elif norm_mean_type == "running":
        reward_mean = running.mean
    elif norm_mean_type == None:
        reward_mean = 0.0
    # 标准差计算
    if norm_std_type == "batch":
        reward_std = response_level_rewards.std()
    elif norm_std_type == "group":
        reward_std = torch.std(reshape_reward, dim=-1, keepdim=True)
    elif norm_std_type == "running":
        reward_std = running.std
    # 选择基础奖励值
    rewards = reshape_reward if norm_mean_type == "group" else response_level_rewards
    # 标准化奖励
    if norm_std_type is not None:
        normalized_rewards = (rewards - reward_mean) / (reward_std + 1e-6)
    else:
        normalized_rewards = (rewards - reward_mean)

    # 如果是对 group mean 归一化，需要恢复原始形状
    if norm_mean_type == "group":
        normalized_rewards = normalized_rewards.reshape(*response_level_rewards.size())
    return normalized_rewards


def difficulty_mask(data: "DataProto", n_sample=-1, low_threshold=0.1, high_threshold=0.95):
    if n_sample > 1:
        scores = data.batch["scores"].clone().detach()
        reshape_score = scores.reshape(*scores.size()[:-1], -1, n_sample)
        reshape_score_mean = reshape_score.mean(dim=-1, keepdim=True).expand_as(reshape_score).reshape(*scores.size())
        data.batch["difficulty_mask"] = (reshape_score_mean > low_threshold) * (reshape_score_mean < high_threshold)
    else:
        data.batch["difficulty_mask"] = torch.ones_like(data.batch["scores"])
    return data


@torch.no_grad()
def compute_token_reward(data: "DataProto", pipeline_config: PPOConfig, kl_ctrl: AdaptiveKLController):
    token_level_rewards = expand_to_token_level(data)
    beta = 0
    kld = compute_approx_kl(
        log_probs=data.batch["old_log_probs"],
        log_probs_base=data.batch["ref_log_probs"],
        action_mask=data.batch["response_mask"][:, 1:],
        kl_penalty=pipeline_config.kl_penalty,
    )
    # 是否添加token level kl
    if pipeline_config.add_token_level_kl and "ref_log_probs" in data.batch.keys():
        beta = kl_ctrl.value
        token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=data.batch["response_mask"][:, 1:], dim=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=data.batch.batch_size[0])
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")
    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=token_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        metrics["critic/token_reward_clip_frac"] = reward_clip_frac
        token_level_rewards = torch.clamp(
            token_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

    data.batch["token_level_rewards"] = token_level_rewards
    return data, metrics


@torch.no_grad()
def reward_postprocess(data: "DataProto", pipeline_config: RLVRConfig, running_ctrl):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    response_level_metrics = {"critic/reward_clip_frac": 0.0}
    # 对reward进行处理: 可以灵活定义不同的normalization方法
    if pipeline_config.adv_estimator == "grpo":
        pipeline_config.norm_mean_type, pipeline_config.norm_std_type = "group", "group"

    response_level_rewards = reward_norm(
                    response_level_rewards,
                    n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
                    running_ctrl=running_ctrl,
                    norm_mean_type=pipeline_config.norm_mean_type,
                    norm_std_type=pipeline_config.norm_std_type
                    )

    # 对reward进行clip
    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=response_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        response_level_rewards = torch.clamp(
            response_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

        response_level_metrics = {"critic/reward_clip_frac": reward_clip_frac}

    data.batch["response_level_rewards"] = response_level_rewards
    return data, response_level_metrics


@torch.no_grad()
def get_sample_level_mask(data: "DataProto", pipeline_config: RLVRConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    # mask相关策略
    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"][:, 1:].clone()
    true_response_length = response_mask.sum(-1).float()
    max_response_length = data.batch["responses"].shape[-1]

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    # 1. max_len_mask: 过滤掉超过最大长度的样本
    if pipeline_config.max_len_mask:
        max_len_mask = (max_response_length != true_response_length).float()
        final_sample_mask = final_sample_mask * max_len_mask
        mask_metrics["actor/max_len_mask_ratio"] = max_len_mask.mean().item()
    else:
        mask_metrics["actor/max_len_mask_ratio"] = 1.0

    # 2. difficulty_mask: 基于难度的过滤
    if pipeline_config.difficulty_mask:
        data = difficulty_mask(
            data,
            n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
            low_threshold=pipeline_config.difficulty_low_threshold,
            high_threshold=pipeline_config.difficulty_high_threshold,
        )
        if "difficulty_mask" in data.batch:
            difficulty_mask_tensor = data.batch["difficulty_mask"].float()
            final_sample_mask = final_sample_mask * difficulty_mask_tensor
            mask_metrics["actor/difficulty_mask_ratio"] = difficulty_mask_tensor.mean().item()
        else:
            mask_metrics["actor/difficulty_mask_ratio"] = 1.0
    else:
        mask_metrics["actor/difficulty_mask_ratio"] = 1.0

    # 3. error_max_len_clip: 基于错误和长度的过滤
    if pipeline_config.error_max_len_clip:
        scores = data.batch["scores"]
        error_len_mask = ((scores == 0) & (true_response_length < pipeline_config.error_max_len_threshold)) | (
            scores == 1
        )
        error_len_mask = error_len_mask.float()
        final_sample_mask = final_sample_mask * error_len_mask
        mask_metrics["actor/error_len_mask_ratio"] = error_len_mask.mean().item()
    else:
        mask_metrics["actor/error_len_mask_ratio"] = 1.0

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["final_response_mask"] = final_response_mask
    return data, mask_metrics


@torch.no_grad()
def apply_kl_penalty(data: "DataProto", kl_ctrl: AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"][:, 1:]

    token_level_rewards = expand_to_token_level(data)
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")

    batch_size = data.batch.batch_size[0]

    if "ref_log_probs" in data.batch.keys():
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=kl_penalty,
        )  # (batch_size, seq_len-1)
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    return data, metrics


@torch.no_grad()
def compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    response_mask=None,
):
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]
    if response_mask.sum() == 0:
        whiten_rewards = False
        whiten_advantages = False
        logger.info("Warning: domain final_response_mask.sum() == 0! All masked_whiten will be skipped.")

    token_level_rewards = data.batch["token_level_rewards"].float()
    if whiten_rewards:
        token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
    token_level_rewards = token_level_rewards * response_mask
    data.batch["token_level_rewards"] = token_level_rewards
    if adv_estimator == "gae":
        values = data.batch["values"].float()
        data.batch["values"] = values * response_mask
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
        )
    elif adv_estimator in ["reinforce", "grpo", "gigpo", "step_reinforce"]:
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    else:
        raise NotImplementedError

    data.batch["raw_advantages"] = advantages
    if whiten_advantages:
        # TODO whiten过程中是否要考虑response的长度？
        advantages = masked_whiten(values=advantages, mask=response_mask)
    advantages = advantages * response_mask

    if advantage_clip is not None:
        adv_clip_frac = compute_clip_fraction(values=advantages, clip_min=-advantage_clip, clip_max=advantage_clip)
        data.meta_info["metrics"] = {"critic/advantage_clip_frac": adv_clip_frac}
        advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

def postprocess_generate(
    prompts: "DataProto",
    output: torch.Tensor,
    num_return_sequences,
    sequence_length,
    eos_token_id,
    pad_token_id,
    fill_eos_token=False,
    output_logprobs: Optional[list[list[float]]] = None,
    pad_to_seq_len=True,
) -> "DataProto":
    from roll.distributed.scheduler.protocol import DataProto

    if fill_eos_token:
        # yali: 如果output最后一个token不是pad_token_id，则替换成eos_token_id,
        #  TODO: 需要消融这个变化的影响
        last_token_index = output.size(1) - 1
        need_replace_mask = output[:, last_token_index] != pad_token_id
        output[need_replace_mask, last_token_index] = eos_token_id

    input_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
    attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
    prompt_id = prompts.batch.get("prompt_id", None)

    # input_batch_size * num_return_sequences
    output_batch_size = output.size(0)
    prompt_length = input_ids.size(1)

    if pad_to_seq_len:
        output = pad_to_length(output, sequence_length, pad_token_id)
        assert output.shape[1] == sequence_length, f"output shape {output.shape} != {sequence_length}"
    sequence_length = output.shape[1]

    prompt = output[:, :prompt_length].clone()  # (bs, prompt_length)
    response = output[:, prompt_length:].clone()  # (bs, response_length)

    attention_mask = (
        attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, prompt_length)
    )
    response_mask = get_pad_mask(
        response_id=response, pad_token=pad_token_id, eos_token=eos_token_id, dtype=attention_mask.dtype
    )
    attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

    position_ids = prompts.batch["position_ids"]
    # if is_num_return_sequences_expand=True, num_return_sequences here equals 1
    if position_ids.dim() == 3:  # qwen2vl mrope, maybe can support in other ways
        position_ids = (
            position_ids.unsqueeze(1)
            .repeat(1, num_return_sequences, 1, 1)
            .view(output_batch_size, *position_ids.shape[-2:])
        )
        delta_position_id = torch.arange(1, (sequence_length - prompt_length) + 1, device=position_ids.device)
        # position_ids: (bsz, C, prompt_len). Expand delta along channel dim (C can be 3 or 4).
        delta_position_id = delta_position_id.view(1, 1, -1).expand(output_batch_size, position_ids.size(1), -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        # left padding for prompt and right padding for response, to be converted
        # to right padding which is consistent with output
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

    assert attention_mask.any(dim=1).all(), f"has all 0 attention_mask, {attention_mask} {input_ids}"
    first_one = attention_mask.float().argmax(dim=1)
    new_response_mask = torch.zeros_like(attention_mask)  # response mask for cat input_ids
    logprobs = (
        torch.zeros([output_batch_size, sequence_length - 1], dtype=torch.float32)
        if output_logprobs is not None
        else None
    )
    for i in range(output_batch_size):
        shift = first_one[i].item()
        if shift > 0:
            output[i, :-shift] = output[i, shift:].clone()
        else:
            output[i, :] = output[i, :].clone()
        valid_length = attention_mask[i].sum().int().item()
        response_length = response_mask[i].sum().int().item()
        attention_mask[i][:valid_length] = 1
        attention_mask[i][valid_length:] = 0
        prompt_len = valid_length - response_length
        new_response_mask[i][prompt_len:valid_length] = 1
        if logprobs is not None:
            logprobs[i][prompt_len - 1 : valid_length - 1] = torch.tensor(
                output_logprobs[i][:response_length], dtype=logprobs.dtype
            )
        if position_ids.dim() == 3 and shift > 0:
            # shift as output to convert to right padding
            # NOTE: left shift without clear right might lead to unclean values
            # in right part, which especially is the case when using long prompt
            # length and short response length. This usually makes no effect if
            # mask is right, while it might make trouble to for multi-modal model
            # like Qwen2-vl, since extra image_token would be left which might
            # cause error: Image features and image tokens do not match
            output_position_ids[i, ..., :-shift] = output_position_ids[i, ..., shift:].clone()
            # only clean in VLM(qwen2-vl) to make no effect on LLM
        if shift > 0 and prompt_length > valid_length:
            output[i, -shift:] = pad_token_id

    prompt_mask = (attention_mask == 1) & (new_response_mask == 0)
    if position_ids.dim() == 3:
        position_ids = output_position_ids
    else:  # normal position_ids
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    batch = TensorDict(
        {
            "prompts": prompt,
            "responses": response,
            "input_ids": output,  # right pad
            "attention_mask": attention_mask,  # right pad
            "position_ids": position_ids,
            "prompt_mask": prompt_mask,
            "response_mask": new_response_mask,  # right pad, response tokens
        },
        batch_size=output_batch_size,
    )
    if prompt_id is not None:
        prompt_id = (
            prompt_id.squeeze().unsqueeze(1).repeat(1, num_return_sequences).view(output_batch_size, -1).squeeze(-1)
        )
        batch["prompt_id"] = prompt_id
    if logprobs is not None:
        batch["infer_logprobs"] = logprobs
    return DataProto(batch=batch)


def get_dist_info_from_comm_plan(comm_plan, rank_in_cluster, rank_in_worker):
    for src_rank, comm_plan_args in comm_plan.items():
        start_rank = 0
        for tgt_device in comm_plan_args["tgt_devices"]:
            start_rank += 1
            if tgt_device["rank"] == rank_in_cluster and tgt_device["device"]["rank"] == rank_in_worker:
                return start_rank, comm_plan_args
    return None, None


def separate_prompt_response(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor, pad_id: int
):
    prompt_mask = attention_mask.bool() & ~response_mask.bool()
    response_mask_valid = attention_mask.bool() & response_mask.bool()
    prompt_ids = torch.where(prompt_mask, input_ids, torch.full_like(input_ids, pad_id))
    response_ids = torch.where(response_mask_valid, input_ids, torch.full_like(input_ids, pad_id))
    return prompt_ids, response_ids

def filter_func_args(func, forward_args):
    signature = inspect.signature(func)
    forward_params = signature.parameters.keys()
    valid_args = {k: v for k, v in forward_args.items() if k in forward_params}
    return valid_args


def aggregate_metrics(history_metrics: List[Dict], metrics_agg_mode: Dict[str, str]) -> Dict[str, float]:
    """
    Aggregate metrics from history based on the specified aggregation modes.

    Args:
        history_metrics: List of dictionaries containing metrics for each step
        metrics_agg_mode: Dictionary mapping metric names to aggregation modes
                         Supported modes: "sum", "mean", "min", "max", "last", "first"

    Returns:
        Dictionary of aggregated metrics
    """
    # Collect all metrics from history
    all_metrics = {}
    for metrics in history_metrics:
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(float(v))

    # Aggregate metrics based on mode
    aggregated_metrics = {}
    for metric_name, values in all_metrics.items():
        mode = metrics_agg_mode.get(metric_name, "mean")  # default to mean
        if mode == "sum":
            aggregated_metrics[metric_name] = float(np.sum(values))
        elif mode == "mean":
            aggregated_metrics[metric_name] = float(np.mean(values))
        elif mode == "min":
            aggregated_metrics[metric_name] = float(np.min(values))
        elif mode == "max":
            aggregated_metrics[metric_name] = float(np.max(values))
        elif mode == "last":
            aggregated_metrics[metric_name] = float(values[-1])
        elif mode == "first":
            aggregated_metrics[metric_name] = float(values[0])
        else:
            # Default to mean for unknown modes
            aggregated_metrics[metric_name] = float(np.mean(values))

    return aggregated_metrics


def group_reward_norm(data: "DataProto", n_sample=-1, div_std=True, div_std_global=False):
    assert n_sample > 1, "n_sample must > 1"
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    reshape_reward = response_level_rewards.reshape(*response_level_rewards.size()[:-1], -1, n_sample)
    reshape_reward = reshape_reward - reshape_reward.mean(dim=-1, keepdim=True)
    if div_std:
        if not div_std_global:
            reshape_reward = reshape_reward / (torch.std(reshape_reward, dim=-1, keepdim=True) + 1e-6)
        else:
            reshape_reward = reshape_reward / (torch.std(reshape_reward) + 1e-6)
    data.batch["response_level_rewards"] = reshape_reward.reshape(*response_level_rewards.size())
    return data


def adjust_sequence_length(sequence, target_length, origin_seq_len, pad_value=0):
    """
    调整序列长度。自动探测序列维度（优先最后一维，其次向前搜索）。

    Args:
        sequence: 输入张量 (e.g., [B, S], [B, S, D], [B, 3, S])
        target_length: 目标的全局序列长度
        origin_seq_len: 当前张量应当对应的参考原始长度
        pad_value: 填充值
    """
    if sequence.dim() < 2:
        return sequence

    # --- 1. 探测序列维度 (seq_dim) ---
    seq_dim = None
    is_causal_shift = False

    # 优先级：最后一维 (-1)，然后是倒数第二维 (-2)，以此类推
    # 检查是否等于参考长度 或 参考长度-1 (causal shift)
    candidate_dims = [-1] + list(range(-2, -sequence.dim() - 1, -1))

    for d in candidate_dims:
        curr_size = sequence.size(d)
        if curr_size == origin_seq_len:
            seq_dim = d
            is_causal_shift = False
            break
        elif curr_size == origin_seq_len - 1:
            seq_dim = d
            is_causal_shift = True
            break

    # 如果没找到任何维度匹配 origin_seq_len，说明该张量不需要处理
    if seq_dim is None:
        return sequence

    # --- 2. 计算实际需要调整到的目标长度 ---
    actual_len = sequence.size(seq_dim)
    # 如果原始是 S-1，目标也应该是 target-1 (保持位移一致)
    effective_target = target_length - 1 if is_causal_shift else target_length

    if actual_len == effective_target:
        return sequence

    # --- 3. 执行 Padding 或 Truncation ---
    if actual_len < effective_target:
        # Padding 逻辑
        pad_size = effective_target - actual_len

        # torch.nn.functional.pad 的 pad 参数顺序是：
        # [最后维左, 最后维右, 倒数第二维左, 倒数第二维右, ...]
        # 我们只在识别到的 seq_dim 的右侧进行 padding
        # 偏移量计算：abs(seq_dim) - 1 决定了前面有多少对 [0, 0]
        pad_config = [0, 0] * (abs(seq_dim) - 1) + [0, pad_size]

        return torch.nn.functional.pad(sequence, pad_config, value=pad_value)

    else:
        # Truncation 逻辑 (通用切片)
        slices = [slice(None)] * sequence.dim()
        slices[seq_dim] = slice(0, effective_target)
        return sequence[tuple(slices)]


def get_seqlen_balanced_partitions(seqlen_list: List[float],
                                   k_partitions: int,
                                   equal_size: bool = False) -> List[List[int]]:
    """
    Reference: https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py

    Partition sequences to balance workload using Karmarkar-Karp algorithm.

    Args:
        seqlen_list: List of sequence lengths (or workloads)
        k_partitions: Number of partitions to create
        equal_size: If True, ensure all partitions have equal number of items

    Returns:
        List of partitions, where each partition is a list of indices
    """

    class Set:
        """Represents a set of items with their sum."""

        def __init__(self):
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: float):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        """Represents a state in the partitioning algorithm."""

        def __init__(self, items: List[Tuple[int, float]], k: int):
            self.k = k
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self) -> List[List[int]]:
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> float:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

    assert len(seqlen_list) >= k_partitions, \
        f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    # Sort by sequence length
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, \
            f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    # Merge states until only one remains
    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()

    # Validate and sort partitions
    assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
    seen_idx = set()
    sorted_partitions = []

    for i, partition in enumerate(partitions):
        assert len(partition) > 0, f"the {i}-th partition is empty"
        for idx in partition:
            seen_idx.add(idx)
        sorted_partitions.append(sorted(partition))

    assert seen_idx == set(range(len(seqlen_list))), "Not all indices are covered"

    return sorted_partitions


def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix):
    """
    Calculate and log metrics related to sequence length imbalance before and after partitioning.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        partitions (List[List[int]]): A list of partitions, where each inner list contains indices
                                      from seqlen_list assigned to that partition.
        prefix (str): A prefix to be added to each metric key in the returned dictionary.

    Returns:
        dict: A dictionary containing metrics related to sequence length imbalance.
    """
    # Get the number of partitions
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0

    # Iterate over each batch of sequence lengths
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset: offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


def batch_balance(batch: DataProto, dp_size, minibatch_size, logging_prefix="global_seqlen", keep_minibatch=False):
    """
    ref: https://github.com/volcengine/verl/blob/2c0fcbe52a9230281329e7197501f4dc67f0a5d8/verl/trainer/ppo/ray_trainer.py#L1018
    Reorder the data on single controller such that each dp rank gets similar total tokens"""
    attention_mask = batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)

    def calculate_workload(seq_len_list):
        return 24576 * seq_len_list + seq_len_list * seq_len_list

    workload_lst = calculate_workload(global_seqlen_lst)
    world_size = dp_size
    if keep_minibatch:
        # Decouple the DP balancing and mini-batching.
        minibatch_num = len(workload_lst) // minibatch_size
        global_partition_lst = [[] for _ in range(world_size)]
        for i in range(minibatch_num):
            rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                workload_lst[i * minibatch_size: (i + 1) * minibatch_size],
                k_partitions=world_size,
                equal_size=True,
            )
            for j, part in enumerate(rearrange_minibatch_lst):
                global_partition_lst[j].extend([x + minibatch_size * i for x in part])
    else:
        global_partition_lst = get_seqlen_balanced_partitions(
            workload_lst, k_partitions=world_size, equal_size=True
        )
    # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
    for idx, partition in enumerate(global_partition_lst):
        partition.sort(key=lambda x: (workload_lst[x], x))
        ordered_partition = partition[::2] + partition[1::2][::-1]
        global_partition_lst[idx] = ordered_partition
    # reorder based on index. The data will be automatically equally partitioned by dispatch function
    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
    batch.reorder(global_idx)
    global_balance_stats = log_seqlen_unbalance(
        seqlen_list=global_seqlen_lst.detach().cpu().tolist(), partitions=global_partition_lst, prefix=logging_prefix
    )
    metrics = {}
    metrics.update(global_balance_stats)
    return metrics

