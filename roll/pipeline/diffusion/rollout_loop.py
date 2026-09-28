"""Diffusion-specific rollout loop.

Subclass of UserDefinedRolloutLoop that provides:
- Deterministic per-sample seed computation
- Unique sample_id injection
- Non-token output schema postprocessing (prompt_embeds alignment, etc.)
- rollout_id / rollout_seed metadata for dump traceability
"""

import hashlib
import uuid
from typing import List

import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.user_defined_rollout_loop import UserDefinedRolloutLoop
from roll.utils.functionals import union_two_dict
from roll.utils.logging import get_logger


logger = get_logger()


def _deterministic_sample_seed(global_seed: int, prompt_id: int, sample_idx: int) -> int:
    """Compute a deterministic per-sample seed from global seed, prompt_id, and sample index.

    Uses a hash to avoid correlated sequences between samples while ensuring
    reproducibility across runs with the same global seed.
    """
    raw = f"{global_seed}-{prompt_id}-{sample_idx}".encode()
    h = int(hashlib.sha256(raw).hexdigest(), 16)
    return h % (2**31 - 1)


class DiffusionRolloutLoop(UserDefinedRolloutLoop):
    """Rollout loop for diffusion (FlowGRPO / NFT) pipelines.

    Overrides prepare_requests and postprocess_output_data to handle
    non-token generation schemas used by diffusion models.
    """

    def prepare_requests(self, request_data_list: List[DataProto], context) -> List[DataProto]:
        """Inject deterministic per-sample seeds and unique sample_ids.

        Args:
            request_data_list: Expanded request list (one per sample).
            context: RolloutContext providing pipeline_config and prompt_id.

        Returns:
            The same list with seed and sample_id injected.
        """
        global_seed = getattr(context.pipeline_config, "seed", 42)

        for sample_idx, req in enumerate(request_data_list):
            # Deterministic seed for reproducibility
            per_sample_seed = _deterministic_sample_seed(global_seed, context.prompt_id, sample_idx)
            req.meta_info["generation_config"]["seed"] = per_sample_seed
            req.meta_info["_request_seed"] = per_sample_seed

            # Unique sample_id: "{prompt_dataset_id}-{sample_idx}"
            prompt_level_id = (
                req.non_tensor_batch["id"][0]
                if "id" in req.non_tensor_batch
                else str(context.prompt_id)
            )
            unique_sample_id = f"{prompt_level_id}-{sample_idx}"
            batch_size = len(req.non_tensor_batch.get("id", [None]))
            req.non_tensor_batch["sample_id"] = np.array(
                [unique_sample_id] * batch_size, dtype=object
            )

        return request_data_list

    def postprocess_output_data(self, request: DataProto, data: DataProto, sequence_length: int) -> DataProto:
        """Non-token output schema postprocess for diffusion models.

        Handles:
        - Extracting tensor/non-tensor fields from meta_info
        - Aligning prompt_embeds / mask dimensions to sequence_length
        - Attaching rollout_id and rollout_seed for traceability
        """
        tensor_fields = {}
        non_tensor_fields = {}
        batch_size = None

        for key in data.meta_info.keys():
            value = data.meta_info[key]
            if isinstance(value, torch.Tensor):
                tensor_fields[key] = value
                if batch_size is None:
                    batch_size = value.shape[0]
            else:
                non_tensor_fields[key] = value

        # Align embedding dimensions to sequence_length
        for key in ["prompt_embeds", "negative_prompt_embeds"]:
            if key not in tensor_fields:
                continue
            v = tensor_fields[key]
            if v.ndim == 2:
                v = v.unsqueeze(0)
            cur_len = v.shape[1]
            if cur_len < sequence_length:
                v = torch.nn.functional.pad(v, (0, 0, 0, sequence_length - cur_len), value=0)
            elif cur_len > sequence_length:
                v = v[:, :sequence_length, :]
            tensor_fields[key] = v

        for key in ["prompt_embeds_mask", "negative_prompt_embeds_mask"]:
            if key not in tensor_fields:
                continue
            v = tensor_fields[key]
            if v.ndim == 1:
                v = v.unsqueeze(0)
            cur_len = v.shape[1]
            if cur_len < sequence_length:
                v = torch.nn.functional.pad(v, (0, sequence_length - cur_len), value=0)
            elif cur_len > sequence_length:
                v = v[:, :sequence_length]
            tensor_fields[key] = v

        # Infer batch_size from non-tensor fields if no tensors found
        if batch_size is None:
            for value in non_tensor_fields.values():
                if isinstance(value, np.ndarray):
                    batch_size = int(value.shape[0]) if value.ndim > 0 else 1
                    break
                if isinstance(value, (list, tuple)):
                    batch_size = len(value)
                    break
            batch_size = batch_size or 1

        # Convert non-tensor fields to object arrays
        non_tensors = {}
        for key, value in non_tensor_fields.items():
            if isinstance(value, np.ndarray) and value.dtype == object and value.ndim > 0 and value.shape[0] == batch_size:
                non_tensors[key] = value
                continue

            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size:
                seq = list(value)
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                seq = list(value)
            else:
                seq = [value] * batch_size

            arr = np.empty(batch_size, dtype=object)
            arr[:] = seq
            non_tensors[key] = arr

        assert tensor_fields, "diffusion postprocess expects tensor outputs (prompt_embeds, etc.)"
        output_data = DataProto.from_dict(
            tensors=tensor_fields,
            non_tensors=dict(request.non_tensor_batch),
            meta_info=dict(request.meta_info),
        )
        for key, value in non_tensors.items():
            if key in output_data.non_tensor_batch:
                continue
            output_data.non_tensor_batch[key] = value
        # Keep batch-aligned tensors in `batch` only. `meta_info` is for scalar/global
        # metadata such as eos/pad ids and generation config.
        output_data.meta_info = union_two_dict(output_data.meta_info, non_tensor_fields)

        # Attach rollout metadata for traceability (used by reward/dump chains)
        req_batch_size = output_data.batch.batch_size[0]
        output_data.non_tensor_batch["rollout_id"] = np.array(
            [str(uuid.uuid4()) for _ in range(req_batch_size)], dtype=object
        )
        gen_seed = output_data.meta_info.get("generation_config", {}).get("seed", None)
        output_data.non_tensor_batch["rollout_seed"] = np.array(
            [gen_seed] * req_batch_size, dtype=object
        )

        # Infer img_shapes from latent spatial dimension for model positional encoding
        if "all_latents" in output_data.batch.keys():
            seq_len = int(output_data.batch["all_latents"].shape[2])
            side = int(seq_len ** 0.5)
            img_shapes = [[(1, side, side)]] if side * side == seq_len else [[(1, seq_len, 1)]]
            output_data.meta_info["img_shapes"] = img_shapes

        return output_data
