import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

from roll.platforms import current_platform
from roll.utils.context_parallel import get_ulysses_group, set_upg_manager
from roll.utils.context_parallel.autograd_gather import ulysses_gather
from roll.utils.context_parallel.rmpad_ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad_and_slice_inputs,
    ulysses_pad_inputs,
)
from roll.utils.fsdp_utils import (
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_init_weight_context_manager,
    get_shard_placement_fn,
)
from roll.utils.functionals import log_probs_from_logits


def _parse_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_lower = dtype.lower()
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
            "fp64": torch.float64,
            "float64": torch.float64,
        }
        if dtype_lower in dtype_map:
            return dtype_map[dtype_lower]
        if hasattr(torch, dtype):
            return getattr(torch, dtype)
        raise ValueError(f"Unsupported dtype string: {dtype}")
    return dtype


def create_device_mesh_with_ulysses(world_size: int, fsdp_size: int):
    """
    Matches `roll.distributed.strategy.fsdp2_strategy.create_device_mesh_with_ulysses`.
    """
    if fsdp_size <= 1 or fsdp_size >= world_size:
        mesh_shape = (world_size,)
        mesh_dim_names = ["fsdp"]
    else:
        ddp_size = world_size // fsdp_size
        mesh_shape = (ddp_size, fsdp_size)
        mesh_dim_names = ["ddp", "fsdp"]
    return init_device_mesh(
        current_platform.device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )


def _validate_ulysses_compat(config, cp_size: int):
    try:
        num_attention_heads, num_key_value_heads = (
            config.num_attention_heads,
            config.num_key_value_heads,
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            config.text_config.num_attention_heads,
            config.text_config.num_key_value_heads,
        )

    assert (
        num_attention_heads % cp_size == 0
    ), f"num_attention_heads {num_attention_heads} must be divisible by ulysses_size {cp_size}"
    assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_size "
        f"{cp_size} or vice versa. Upon ulysses_size % num_key_value_heads == 0, "
        f"kv heads are repeated to ensure correctness."
    )


@dataclass
class StandaloneRankInfo:
    dp_rank: int
    dp_size: int
    cp_rank: int
    cp_size: int


@dataclass
class StandaloneFSDP2Config:
    model_name_or_path: str
    is_trainable: bool = False
    # FSDP2
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    reshard_after_forward: bool = True
    fsdp_size: int = 1
    cpu_offload: bool = False
    # CP(Ulysses)
    ulysses_size: int = 1
    use_remove_padding: bool = False
    # HF
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None  # e.g. "fa2" / "sdpa" / None


class StandaloneFSDP2Strategy:
    def __init__(self, cfg: StandaloneFSDP2Config):
        self.cfg = cfg
        self.rank_info: Optional[StandaloneRankInfo] = None
        self.device_mesh = None
        self.fsdp_config: Optional[Dict[str, Any]] = None
        self.model: Optional[torch.nn.Module] = None
        self.config = None
        self.param_dtype = _parse_dtype(cfg.param_dtype)
        self.reduce_dtype = _parse_dtype(cfg.reduce_dtype)

    def _init_dist_if_needed(self):
        if dist.is_initialized():
            return
        if current_platform.device_type != "cpu":
            backends_str = f"cpu:gloo,{current_platform.device_type}:{current_platform.communication_backend}"
        else:
            backends_str = current_platform.communication_backend
        dist.init_process_group(backend=backends_str)

    def _setup_rank_info(self) -> StandaloneRankInfo:
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        cp_size = int(self.cfg.ulysses_size or 1)
        if cp_size > 1:
            patch_info = current_platform.apply_ulysses_patch()
            if patch_info is None:
                cp_size = 1

        dp_rank = global_rank // cp_size
        dp_size = world_size // cp_size
        cp_rank = global_rank % cp_size

        info = StandaloneRankInfo(dp_rank=dp_rank, dp_size=dp_size, cp_rank=cp_rank, cp_size=cp_size)
        self.rank_info = info
        return info

    def _setup_device(self):
        if current_platform.device_type == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", str(dist.get_rank())))
            torch.cuda.set_device(local_rank)

    def setup_fsdp2_configuration(self):
        mixed_precision = MixedPrecisionPolicy(
            param_dtype=self.param_dtype,
            reduce_dtype=self.reduce_dtype,
            cast_forward_inputs=True,
        )

        offload_policy = None
        if bool(self.cfg.cpu_offload):
            offload_policy = CPUOffloadPolicy(pin_memory=True)

        self.fsdp_config = {
            "mesh": self.device_mesh,
            "reshard_after_forward": bool(self.cfg.reshard_after_forward),
            "mp_policy": mixed_precision,
            "offload_policy": offload_policy,
            "shard_placement_fn": get_shard_placement_fn(fsdp_size=int(self.cfg.fsdp_size or 1)),
        }

    def _pick_model_class(self, cfg) -> Any:
        if type(cfg) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
            return AutoModelForVision2Seq
        return AutoModelForCausalLM

    def _apply_roll_model_patches(self, model: torch.nn.Module, cfg) -> None:
        # Mirror the important parts of `roll.models.model_providers.load_model` that affect CP/FSDP2.
        model_type = getattr(cfg, "model_type", None) or ""
        ulysses_size = int(self.rank_info.cp_size if self.rank_info is not None else 1)
        # Apply the same shared model forward patches as the main codebase.
        from roll.models.model_providers import patch_model

        patch_model(model, cfg, use_mcore=False)

        if ulysses_size > 1 and getattr(cfg, "vision_config", None) is not None:
            if model_type in ("qwen2_5_vl", "qwen3_vl"):
                from roll.utils.context_parallel.vlm_cp_patch import find_vlm_text_decoder, patch_vlm_decoder_for_cp

                decoder = find_vlm_text_decoder(model)
                if decoder is not None:
                    patch_vlm_decoder_for_cp(decoder, name=f"{model_type}.text_decoder")

        if getattr(cfg, "vision_config", None) is not None:
            # Ensure vision tower blocks do not cast forward inputs under FSDP2.
            from roll.models.model_providers import get_vl_model_vision_tower_blocks

            vision_tower_blocks = get_vl_model_vision_tower_blocks(model)
            if vision_tower_blocks is not None:
                for block in vision_tower_blocks:
                    block._fsdp2_cast_forward_inputs = False

    def initialize(self):
        self._init_dist_if_needed()
        self._setup_device()
        info = self._setup_rank_info()

        world_size = dist.get_world_size()

        fsdp_size = int(self.cfg.fsdp_size or 1)
        if info.cp_size > 1 and (fsdp_size <= 1 or fsdp_size >= world_size):
            fsdp_size = world_size // info.cp_size
            self.cfg.fsdp_size = fsdp_size

        if info.cp_size > 1:
            set_upg_manager(ulysses_size=info.cp_size, rank=dist.get_rank(), world_size=world_size)

        self.device_mesh = create_device_mesh_with_ulysses(world_size=world_size, fsdp_size=fsdp_size)

        hf_cfg = AutoConfig.from_pretrained(self.cfg.model_name_or_path, trust_remote_code=self.cfg.trust_remote_code)
        self.config = hf_cfg
        if info.cp_size > 1:
            _validate_ulysses_compat(hf_cfg, info.cp_size)

        if getattr(hf_cfg, "vision_config", None) is not None:
            vc = hf_cfg.vision_config
            setattr(vc, "_attn_implementation", "sdpa")
            setattr(vc, "attn_implementation", "sdpa")

        setattr(hf_cfg, "use_cache", not bool(self.cfg.is_trainable))

        use_meta_tensor = not getattr(hf_cfg, "tie_word_embeddings", False)
        init_context = get_init_weight_context_manager(use_meta_tensor=use_meta_tensor, mesh=self.device_mesh)

        model_cls = self._pick_model_class(hf_cfg)
        with init_context():
            model = model_cls.from_pretrained(
                self.cfg.model_name_or_path,
                config=hf_cfg,
                trust_remote_code=self.cfg.trust_remote_code,
                low_cpu_mem_usage=False,
            )

        self._apply_roll_model_patches(model, hf_cfg)
        is_lora = getattr(model, "peft_config", None) is not None

        full_state = model.state_dict()

        self.setup_fsdp2_configuration()
        assert self.fsdp_config is not None
        # `apply_fsdp2()` needs a wrap policy list. Most HF models expose `_no_split_modules`,
        # but some custom models may not; fall back to a conservative module-level wrap.
        wrap_list = getattr(model, "_no_split_modules", None)
        if not wrap_list:
            wrap_list = ["Linear"]
        strategy_cfg = {"wrap_policy": {"transformer_layer_cls_to_wrap": wrap_list}}
        apply_fsdp2(model, self.fsdp_config, config=strategy_cfg, is_lora=is_lora)

        fsdp2_load_full_state_dict(
            model=model,
            full_state=full_state,
            device_mesh=self.device_mesh,
            cpu_offload=self.fsdp_config["offload_policy"],
        )

        self.model = model
        dist.barrier()

    def unwrap_model(self):
        if self.model is None:
            return None
        return getattr(self.model, "module", self.model)

    def get_feature_on_cp_rank(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        assert self.rank_info is not None
        seqlens_in_batch = input_ids.size(1)
        assert (
            seqlens_in_batch % self.rank_info.cp_size == 0
        ), f"input_length={seqlens_in_batch} not divisible by cp_size={self.rank_info.cp_size}"
        cp_middle_rank_len = seqlens_in_batch // self.rank_info.cp_size
        start_index = cp_middle_rank_len * self.rank_info.cp_rank
        end_index = cp_middle_rank_len * (self.rank_info.cp_rank + 1)

        result = {"input_ids": input_ids[:, start_index:end_index]}
        if attention_mask is not None:
            result["attention_mask"] = attention_mask[:, start_index:end_index]
        if position_ids is not None:
            if position_ids.dim() == 3:
                result["position_ids"] = position_ids[:, :, start_index:end_index]
            else:
                result["position_ids"] = position_ids[:, start_index:end_index]
        return result

    def fsdp2_forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        forward_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Mirrors `FSDP2InferStrategy._fsdp2_forward`.
        Returns logits (possibly CP-sliced then gathered/padded back to keep downstream shape consistent).
        """
        assert self.model is not None
        assert self.rank_info is not None
        forward_args = dict(forward_args or {})

        cp_size = self.rank_info.cp_size
        cp_rank = self.rank_info.cp_rank

        underlying = self.unwrap_model()
        model_type = getattr(getattr(underlying, "config", None), "model_type", "") or ""
        is_vlm = getattr(getattr(underlying, "config", None), "vision_config", None) is not None
        is_supported_vlm = is_vlm and model_type in ("qwen2_5_vl", "qwen2_vl", "qwen3_vl", "qwen3_vl_moe")

        if "use_cache" not in forward_args:
            forward_args["use_cache"] = False

        # Remove padding + CP path
        if cp_size > 1 and self.cfg.use_remove_padding:
            try:
                from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
            except Exception as e:
                raise RuntimeError("use_remove_padding=True requires flash_attn installed.") from e

            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            if position_ids is None:
                raise RuntimeError("remove_padding path requires position_ids.")

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(
                        rearrange(position_ids, "c b s ... -> (b s) c ..."),
                        indices,
                    )
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (C, 1, total_nnz)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

            if is_supported_vlm:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_inputs(
                    input_ids_rmpad,
                    position_ids_rmpad,
                    cp_size=cp_size,
                )
            else:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad,
                    position_ids_rmpad,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                )

            output = self.model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **forward_args,
            )
            logits_rmpad = output.logits  # (1, local_tokens, vocab)

            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad,
                gather_dim=1,
                unpad_dim=1,
                padding_size=pad_size,
                group=get_ulysses_group(),
            )

            logits = pad_input(
                hidden_states=logits_rmpad.squeeze(0).unsqueeze(-1),
                indices=indices,
                batch=input_ids.size(0),
                seqlen=input_ids.size(1),
            ).squeeze(-1)

            features = self.get_feature_on_cp_rank(logits)
            return features["input_ids"]

        # CP slicing path (non-rmpad)
        if cp_size > 1 and (not is_supported_vlm):
            feats = self.get_feature_on_cp_rank(input_ids, attention_mask, position_ids)
            input_ids = feats["input_ids"]
            attention_mask = feats["attention_mask"]
            position_ids = feats["position_ids"]

        if not self.cfg.use_remove_padding:
            if cp_size > 1 and is_supported_vlm:
                assert (
                    input_ids.size(1) % cp_size == 0
                ), f"input_length={input_ids.size(1)} not divisible by cp_size={cp_size} for VLM non-rmpad CP"
                logits_local = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **forward_args,
                ).logits  # (bs, local_seq, vocab)
                logits_full = gather_outputs_and_unpad(
                    logits_local,
                    gather_dim=1,
                    unpad_dim=None,
                    padding_size=0,
                    group=get_ulysses_group(),
                )
                features = self.get_feature_on_cp_rank(logits_full)
                return features["input_ids"]

            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **forward_args,
            ).logits

        # remove-padding without CP (or cp_size==1)
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except Exception as e:
            raise RuntimeError("use_remove_padding=True requires flash_attn installed.") from e

        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        if position_ids is None:
            raise RuntimeError("remove_padding path requires position_ids.")

        if position_ids.dim() == 3:
            position_ids_rmpad = (
                index_first_axis(
                    rearrange(position_ids, "c b s ... -> (b s) c ..."),
                    indices,
                )
                .transpose(0, 1)
                .unsqueeze(1)
            )
        else:
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                indices,
            ).transpose(0, 1)

        output = self.model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids_rmpad,
            **forward_args,
        )
        logits = pad_input(
            hidden_states=output.logits.squeeze(0).unsqueeze(-1),
            indices=indices,
            batch=input_ids.size(0),
            seqlen=input_ids.size(1),
        ).squeeze(-1)
        return logits

    def compute_log_probs(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mirrors `FSDP2InferStrategy.op_compute_log_probs`.
        Returns per-token logprobs aligned to `attention_mask[:, 1:]` (shifted labels).
        """
        assert self.rank_info is not None

        labels = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = 0

        if self.rank_info.cp_size > 1:
            labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
            labels = self.get_feature_on_cp_rank(labels)["input_ids"]

            log_probs = log_probs_from_logits(logits, labels)
            log_probs = ulysses_gather(
                log_probs,
                gather_dim=1,
                group=get_ulysses_group(),
                grad_scaler=True,
            )
            log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
        else:
            labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
            log_probs = log_probs_from_logits(logits, labels)
            log_probs = log_probs[:, :-1] * attention_mask[:, 1:]

        return log_probs

    @contextlib.contextmanager
    def autocast(self):
        if current_platform.device_type == "cpu":
            yield
            return
        with torch.autocast(device_type=current_platform.device_type, dtype=self.param_dtype):
            yield
