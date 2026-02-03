import inspect
import os
import threading
from typing import Any, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.dynamic_module_utils import get_cached_module_file
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from roll.configs import ModelArguments
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import download_model, file_lock_context
from roll.utils.logging import get_logger
from roll.utils.packages import is_transformers_version_greater_than

try:
    from mcore_adapter import TrainingArguments as mca_TrainingArguments
    from mcore_adapter.adapters import (
        apply_megatron_lora,
        find_all_embedding_modules,
        find_all_linear_modules,
        find_all_router_modules,
        set_linear_is_expert,
    )
    from mcore_adapter.models import AutoModel
except Exception as e:
    mca_TrainingArguments = None


logger = get_logger()

# Thread-local storage for FSDP2 initialization context
_fsdp2_init_context = threading.local()


def set_fsdp2_init_context(context):
    _fsdp2_init_context.context = context


def clear_fsdp2_init_context():
    if hasattr(_fsdp2_init_context, "context"):
        delattr(_fsdp2_init_context, "context")


def is_fsdp2_enabled():
    if hasattr(_fsdp2_init_context, "context"):
        return True
    return False


def is_fsdp_or_fsdp2_enabled():
    return is_fsdp_enabled() or is_fsdp2_enabled()


def prepare_automap_files(model_path: str):
    python_files = []
    for file_name in os.listdir(model_path):
        if file_name.endswith(".py") and os.path.isfile(os.path.join(model_path, file_name)):
            python_files.append(file_name)
    with file_lock_context(model_path):
        for file_name in python_files:
            try:
                get_cached_module_file(model_path, file_name)
            except Exception:
                # if it's a needed file, will raise an exception when calling from_pretrained
                pass


def default_tokenizer_provider(model_args: "ModelArguments", model_name_or_path: str = None):
    if model_args.model_type == "diffusion_module":
        return None
    if model_name_or_path is None:
        model_name_or_path = model_args.model_name_or_path
    model_name_or_path = download_model(model_name_or_path)
    prepare_automap_files(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        split_special_tokens=False,
        trust_remote_code=True,
        padding_side="left",
    )
    return tokenizer


def default_processor_provider(model_args: "ModelArguments", model_name_or_path: str = None):
    if model_args.model_type == "diffusion_module":
        return None
    if model_name_or_path is None:
        model_name_or_path = model_args.model_name_or_path
    model_name_or_path = download_model(model_args.model_name_or_path)
    prepare_automap_files(model_name_or_path)
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        logger.info(f"processor not found: {e}")
        processor = None
    return processor


def load_valuehead_params(model_path):
    """
    modified from llamafactory
    """
    err_text = ""

    try:
        from safetensors import safe_open

        vhead_file = os.path.join(model_path, "value_head.safetensors")
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = os.path.join(model_path, "value_head.bin")
        return torch.load(vhead_file, map_location="cpu")
    except Exception as err:
        err_text = str(err)

    logger.info("Provided path ({}) does not contain value head weights: {}.".format(model_path, err_text))
    logger.info("Ignore the above message if you are not resuming the training of a value head model.")
    return None


def freeze_model(model, model_args: "ModelArguments"):
    if model_args.freeze_module_prefix is None:
        return

    prefixes = model_args.freeze_module_prefix
    logger.info(f"Freeze model with prefix: {prefixes}")
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(False)


# Inspired by: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/adapter.py
def setup_lora_training(
    config, model, model_args: "ModelArguments", is_trainable: Optional[bool] = False, is_mca: Optional[bool] = False
):
    model.enable_input_require_grads()
    if is_trainable:

        def get_target_modules(model: "torch.nn.Module", model_args: "ModelArguments"):
            target_modules = model_args.lora_target
            if "all-linear" in model_args.lora_target:
                target_modules.remove("all-linear")
                target_modules += find_all_linear_modules(model)
            if "all-embedding" in model_args.lora_target:
                target_modules.remove("all-embedding")
                target_modules += find_all_embedding_modules(model)
            if "all-router" in model_args.lora_target:
                target_modules.remove("all-router")
                target_modules += find_all_router_modules(model)
            return target_modules

        target_modules = get_target_modules(model, model_args)
        lora_config = {
            "r": model_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
            "modules_to_save": model_args.additional_target,
        }
        if not is_mca:
            lora_config.update({"task_type": TaskType.CAUSAL_LM})
        model = get_peft_model(
            model, LoraConfig(**lora_config), autocast_adapter_dtype=model_args.autocast_adapter_dtype
        )
    return model


def load_model(
    model_args: "ModelArguments",
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False,
):
    r"""
    modified from llamafactory
    """
    model_name_or_path = download_model(model_args.model_name_or_path)
    prepare_automap_files(model_args.model_name_or_path)
    init_kwargs = {
        "trust_remote_code": True,
        **model_args.model_config_kwargs,
    }
    config = AutoConfig.from_pretrained(model_name_or_path, **init_kwargs)
    if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
        setattr(config, "_attn_implementation", model_args.attn_implementation)

    # ---------------------------------------------------------------------
    # PumpkinComment:
    # Ref: https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py
    # Many VLMs have a separate vision attention stack. When Ulysses/CP is enabled, we patch
    # HF flash-attention paths for text attention. However, vision attention often:
    # - does not carry the same kwargs (e.g. missing position_ids), or
    # - calls into different flash-attn wrappers,
    # which can cause mismatched collectives / deadlocks across ranks.
    ulysses_size = int(model_args.ulysses_size or 1)
    if getattr(config, "vision_config", None) is not None:
        vc = config.vision_config
        setattr(vc, "_attn_implementation", "sdpa")
        setattr(vc, "attn_implementation", "sdpa")

    if not is_trainable:
        setattr(config, "use_cache", True)
    else:
        setattr(config, "use_cache", False)
    if model_args.moe_aux_loss_coef is not None:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)
        setattr(config, "output_router_logits", is_trainable)
    init_kwargs["low_cpu_mem_usage"] = not is_deepspeed_zero3_enabled() and not is_fsdp2_enabled()
    if not is_deepspeed_zero3_enabled() and not is_fsdp_or_fsdp2_enabled():
        init_kwargs["torch_dtype"] = model_args.compute_dtype
        if init_kwargs["low_cpu_mem_usage"]:  # device map requires low_cpu_mem_usage=True
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_name_or_path
    if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
        model_class = AutoModelForVision2Seq  # image and video
    else:
        model_class = AutoModelForCausalLM  # text

    fsdp2_init_context = getattr(_fsdp2_init_context, "context", None)

    if fsdp2_init_context is not None:
        with fsdp2_init_context():
            model = model_class.from_pretrained(**init_kwargs)
    else:
        model = model_class.from_pretrained(**init_kwargs)

    if not model_args.disable_gradient_checkpointing:
        # PumpkinComment:
        # - use_reentrant=False is generally preferred, but some MoE models can produce
        #   a different set of autograd-saved tensors between forward and recomputation,
        #   which triggers torch.utils.checkpoint.CheckpointError.
        if model_args.gradient_checkpointing_use_reentrant is None:
            use_reentrant = True if _is_moe_config(config) else False
        else:
            use_reentrant = bool(model_args.gradient_checkpointing_use_reentrant)

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})

    if model_args.lora_target is None:
        freeze_model(model, model_args)
    else:
        model = setup_lora_training(config, model, model_args, is_trainable)

    if add_valuehead:
        from trl import AutoModelForCausalLMWithValueHead

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model, **model_args.model_config_kwargs)

        vhead_params = load_valuehead_params(model_name_or_path)
        if vhead_params is not None:
            if is_deepspeed_zero3_enabled():
                import deepspeed  # type: ignore

                params = [param for _, param in model.v_head.named_parameters(recurse=False)]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        model.load_state_dict(vhead_params, strict=False)
            else:
                model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(model_name_or_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    # Debug case, we may not use the default_actor model provider
    patch_model(model, config, use_mcore=False)

    if ulysses_size > 1 and getattr(config, "vision_config", None) is not None:
        model_type = getattr(config, "model_type", None) or ""
        if model_type in ("qwen2_5_vl", "qwen3_vl"):
            from roll.utils.context_parallel.vlm_cp_patch import patch_vlm_decoder_for_cp

            decoder = find_vlm_text_decoder(model)
            if decoder is None:
                logger.warning(f"CP(VLM) enabled but failed to locate text decoder for model_type={model_type}")
            else:
                patch_vlm_decoder_for_cp(decoder, name=f"{model_type}.text_decoder")
        else:
            logger.warning(f"CP(VLM) enabled but model_type={model_type} not fully tested")

    if is_fsdp2_enabled() and getattr(config, "vision_config", None) is not None:
        # PumpkinComment:
        # Otherwise we will have precision issue
        vision_tower_blocks = get_vl_model_vision_tower_blocks(model)
        if vision_tower_blocks is not None:
            for block in vision_tower_blocks:
                block._fsdp2_cast_forward_inputs = False

    return model


def patch_model(model, config, use_mcore):
    import types

    model_type = config.model_type

    # Avoid double-patching when multiple providers call patch_model()
    if getattr(model, "_roll_forward_patched", False):
        return

    forward_patch = None
    # patch to force vit forward with mock image to avoid hang
    if not use_mcore:
        if model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
            if is_peft_model := (getattr(model, "peft_config", None) is not None):
                ori_forward = type(model.get_base_model()).forward
            else:
                ori_forward = type(model).forward

            def _handle_missing_visual(self, inputs_embeds: "torch.FloatTensor"):
                if getattr(self.config, "model_type", None) == "qwen3_vl":
                    # Qwen3-VL vision forward returns (image_embeds, deepstack_embeds_list)
                    patch_dim = (
                        self.config.vision_config.in_channels
                        * self.config.vision_config.temporal_patch_size
                        * self.config.vision_config.patch_size
                        * self.config.vision_config.patch_size
                    )
                    mock_pixel_values = torch.zeros(
                        16,
                        patch_dim,
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                    mock_grid_thw = torch.LongTensor([[1, 4, 4]]).to(inputs_embeds.device)
                    vision_out = self.visual(mock_pixel_values, grid_thw=mock_grid_thw)
                    image_embeddings = vision_out[0] if isinstance(vision_out, tuple) else vision_out
                    deepstack_list = vision_out[1] if isinstance(vision_out, tuple) and len(vision_out) > 1 else []
                    inputs_embeds = inputs_embeds + image_embeddings.mean() * 0
                    for emb in deepstack_list or []:
                        inputs_embeds = inputs_embeds + emb.mean() * 0
                    return inputs_embeds

                mock_pixel_values = torch.zeros(
                    4,
                    self.config.vision_config.in_channels
                    * self.config.vision_config.temporal_patch_size
                    * self.config.vision_config.patch_size
                    * self.config.vision_config.patch_size,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                mock_grid_thw = torch.LongTensor([[1, 2, 2]]).to(inputs_embeds.device)
                image_embeddings = self.visual(mock_pixel_values, grid_thw=mock_grid_thw)
                inputs_embeds = inputs_embeds + image_embeddings.mean() * 0
                return inputs_embeds

            def forward_patch(
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
            ):
                assert inputs_embeds is None
                if kwargs.pop("force_vit_image", False) and pixel_values is None:
                    # force vit forward with mock image to avoid hang
                    inputs_embeds = self.get_input_embeddings()(input_ids)
                    inputs_embeds = _handle_missing_visual(self, inputs_embeds)
                if kwargs.pop("force_vit_video", False) and pixel_values_videos is None:
                    if inputs_embeds is None:
                        inputs_embeds = self.get_input_embeddings()(input_ids)
                    # force vit forward with mock image to avoid hang
                    inputs_embeds = _handle_missing_visual(self, inputs_embeds)
                return ori_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    **kwargs,
                )

        if forward_patch is not None:
            if is_peft_model:
                model.get_base_model().forward = types.MethodType(forward_patch, model.get_base_model())
            else:
                model.forward = types.MethodType(forward_patch, model)
            setattr(model, "_roll_forward_patched", True)


def default_diffusion_module_provider(
    tokenizer: None,
    model_args: ModelArguments,
    training_args: TrainingArguments = None,
    is_trainable: Optional[bool] = False,
):
    if model_args.model_config_kwargs["model_name"] == "wan2_2":
        from roll.pipeline.diffusion.modules.wan_module import WanTrainingModule

        print(f"{model_args.model_config_kwargs=}")
        training_module = WanTrainingModule(**model_args.model_config_kwargs)
    else:
        raise NotImplementedError(f"model_type {model_args.model_type} not implemented yet")

    return training_module


def default_actor_model_provider(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    training_args: "TrainingArguments" = None,
    is_trainable: Optional[bool] = False,
):
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    old_model_name_or_path = model_args.model_name_or_path
    prepare_automap_files(model_args.model_name_or_path)
    if (
        mca_TrainingArguments is not None
        and training_args is not None
        and isinstance(training_args, mca_TrainingArguments)
    ):
        # megatron
        if model_args.moe_aux_loss_coef is not None and training_args.moe_aux_loss_coeff is None:
            training_args.moe_aux_loss_coeff = model_args.moe_aux_loss_coef
        model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
        if is_trainable:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        if model_args.lora_target is None:
            freeze_model(model, model_args)
        else:
            apply_megatron_lora()
            set_linear_is_expert(model[0])
            model.models[0] = setup_lora_training(model[0].config, model[0], model_args, is_trainable, is_mca=True)
        patch_model(model, config, use_mcore=True)
    else:
        # hf
        init_kwargs = {
            "torch_dtype": model_args.compute_dtype,
            "trust_remote_code": True,
        }
        if not is_deepspeed_zero3_enabled() and not is_fsdp2_enabled():
            init_kwargs["low_cpu_mem_usage"] = True
            if is_trainable:
                init_kwargs["device_map"] = {"": current_platform.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
            elif model_args.export_dir is None:
                init_kwargs["device_map"] = "balanced"
        logger.info(f"init_kwargs: {init_kwargs}")
        model = load_model(model_args, is_trainable, False)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        patch_model(model, config, use_mcore=False)

    model_args.model_name_or_path = old_model_name_or_path
    return model


def default_reward_model_provider(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    training_args: "TrainingArguments" = None,
    is_trainable: Optional[bool] = False,
):
    """
    model.forward 遵循TokenClassifierOutput 协议
    class TokenClassifierOutput(ModelOutput):
        logits: torch.FloatTensor   # 必须要有
        loss: Optional[torch.FloatTensor] = None
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    """
    old_model_name_or_path = model_args.model_name_or_path
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)
    prepare_automap_files(model_args.model_name_or_path)

    if (
        mca_TrainingArguments is not None
        and training_args is not None
        and isinstance(training_args, mca_TrainingArguments)
    ):
        # megatron
        raise NotImplementedError("megatron reward model not implemented")
    else:
        init_kwargs = {
            "torch_dtype": model_args.compute_dtype,
            "trust_remote_code": True,
        }
        if not is_deepspeed_zero3_enabled():
            init_kwargs["low_cpu_mem_usage"] = True
            if is_trainable:
                init_kwargs["device_map"] = {"": current_platform.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
        logger.info(f"init_kwargs: {init_kwargs}")
        if model_args.model_type in ["auto_sequence_classification"]:
            logger.info(f"use AutoModelForSequenceClassification model {model_args.model_type}")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.num_labels = model_args.num_labels
            if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
                setattr(
                    config,
                    "_attn_implementation",
                    model_args.attn_implementation,
                )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path, config=config, **init_kwargs
            )
        elif model_args.model_type in ["auto_token_classification"]:
            logger.info(f"use AutoModelForTokenClassification model {model_args.model_type}")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.num_labels = model_args.num_labels
            if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
                setattr(
                    config,
                    "_attn_implementation",
                    model_args.attn_implementation,
                )
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path, config=config, **init_kwargs
            )
        elif model_args.model_type in ["trl"]:
            from trl import AutoModelForCausalLMWithValueHead

            from roll.models.trl_patches import (
                no_set_device_hook_post_init,
                token_classifier_forward,
                value_head_load_state_dict,
            )

            AutoModelForCausalLMWithValueHead.post_init = no_set_device_hook_post_init
            model = load_model(model_args, is_trainable, True)
            setattr(model, "forward", token_classifier_forward.__get__(model))
            setattr(
                model,
                "load_state_dict",
                value_head_load_state_dict.__get__(model),
            )
            logger.info("patch AutoModelForCausalLMWithValueHead load_state_dict and forward")
        else:
            raise NotImplementedError
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model_args.model_name_or_path = old_model_name_or_path

    return model


def default_value_model_provider(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    training_args: "TrainingArguments" = None,
    is_trainable: Optional[bool] = False,
):
    """
    TokenClassifierOutput
    """
    old_model_name_or_path = model_args.model_name_or_path
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)
    prepare_automap_files(model_args.model_name_or_path)

    if (
        mca_TrainingArguments is not None
        and training_args is not None
        and isinstance(training_args, mca_TrainingArguments)
    ):
        raise NotImplementedError("megatron value model not implemented")
    else:
        init_kwargs = {
            "torch_dtype": model_args.compute_dtype,
            "trust_remote_code": True,
        }
        if not is_deepspeed_zero3_enabled():
            init_kwargs["low_cpu_mem_usage"] = True
            if is_trainable:
                init_kwargs["device_map"] = {"": current_platform.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
        logger.info(f"init_kwargs: {init_kwargs}")
        if model_args.model_type in ["auto_token_classification"]:
            logger.info(f"use AutoModelForTokenClassification model {model_args.model_type}")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.num_labels = model_args.num_labels
            if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
                setattr(
                    config,
                    "_attn_implementation",
                    model_args.attn_implementation,
                )
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path, config=config, **init_kwargs
            )
        elif model_args.model_type in ["trl"]:
            from trl import AutoModelForCausalLMWithValueHead

            from roll.models.trl_patches import (
                no_set_device_hook_post_init,
                token_classifier_forward,
                value_head_load_state_dict,
            )

            AutoModelForCausalLMWithValueHead.post_init = no_set_device_hook_post_init
            model = load_model(model_args, is_trainable, True)
            setattr(model, "forward", token_classifier_forward.__get__(model))
            setattr(
                model,
                "load_state_dict",
                value_head_load_state_dict.__get__(model),
            )
        else:
            raise NotImplementedError
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model_args.model_name_or_path = old_model_name_or_path

    return model


def get_extra_data_provider(model_name_or_path: str, processor=None):
    model_name_or_path = download_model(model_name_or_path)
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = config.model_type
    except ValueError as e:
        # mca ckpt use mca_config.json as config file
        import json

        from mcore_adapter.constants import MCA_CONFIG_NAME

        config_file = os.path.join(model_name_or_path, MCA_CONFIG_NAME)
        model_type = None
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_values = json.loads(text)
            model_type = config_values.get("hf_model_type")
        else:
            raise e

    # NOTE:
    if isinstance(model_type, str) and (("qwen2" in model_type) or (model_type in ("qwen3_vl", "qwen3_vl_moe"))):
        import types

        from transformers import BatchFeature  # help define a object to accesss attr

        def _call_get_rope_index(fn, input_ids: torch.LongTensor, **candidate_kwargs):
            sig = inspect.signature(fn)
            params = sig.parameters
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if accepts_kwargs:
                return fn(input_ids, **candidate_kwargs)

            filtered = {k: v for k, v in candidate_kwargs.items() if k in params}
            return fn(input_ids, **filtered)

        spatial_merge_size = getattr(getattr(config, "vision_config", None), "spatial_merge_size", None)
        if spatial_merge_size is None and processor is not None:
            spatial_merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", None)
        if spatial_merge_size is None:
            raise ValueError(
                f"spatial_merge_size is required for model_type={model_type} get_rope_index, "
                "but it was not found in config.vision_config nor processor.image_processor."
            )
        vc = {"spatial_merge_size": spatial_merge_size}
        tokens_per_second = getattr(getattr(config, "vision_config", None), "tokens_per_second", None)
        if model_type == "qwen2_5_vl" and tokens_per_second is not None:
            vc["tokens_per_second"] = tokens_per_second

        image_token_id = getattr(config, "image_token_id", None)
        video_token_id = getattr(config, "video_token_id", None)
        vision_start_token_id = getattr(config, "vision_start_token_id", None)
        if processor is not None and hasattr(processor, "tokenizer"):
            image_token_id = image_token_id or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            video_token_id = video_token_id or processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            vision_start_token_id = vision_start_token_id or processor.tokenizer.convert_tokens_to_ids(
                "<|vision_start|>"
            )

        dummy_self = BatchFeature(
            {
                "config": BatchFeature(
                    {
                        "vision_config": BatchFeature(vc),
                        "image_token_id": image_token_id,
                        "video_token_id": video_token_id,
                        "vision_start_token_id": vision_start_token_id,
                    }
                )
            }
        )

        is_tf_ge_4_52 = is_transformers_version_greater_than("4.52.0")
        if model_type == "qwen2_5_vl":
            if is_tf_ge_4_52:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

                get_rope_index = types.MethodType(Qwen2_5_VLModel.get_rope_index, dummy_self)
            else:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

                get_rope_index = types.MethodType(Qwen2_5_VLForConditionalGeneration.get_rope_index, dummy_self)
        elif model_type in ("qwen3_vl", "qwen3_vl_moe"):
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

            get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, dummy_self)
        else:
            if is_tf_ge_4_52:
                from transformers.models.qwen2_vl import Qwen2VLModel

                get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, dummy_self)
            else:
                from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

                get_rope_index = types.MethodType(Qwen2VLForConditionalGeneration.get_rope_index, dummy_self)

        def extra_data_provider(
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
        ):
            # Keep kwargs to be resilient to HF signature changes between versions/models.
            out = _call_get_rope_index(
                get_rope_index,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            rope_index = out[0]
            # PumpkinComment:
            # HF Qwen-VL "mrope" position_ids are expected to be 4-channel in newer transformers:
            #   [text_pos_ids, mrope_t, mrope_h, mrope_w]
            # while some HF get_rope_index implementations return only the 3D vision part (3, bsz, seqlen).
            #
            # I normalize here so downstream strategies don't need model-specific hacks.
            # Note: transformers < 4.54 only accepts vision position ids in some Qwen-VL variants.
            if is_transformers_version_greater_than("4.53.3") and rope_index.dim() == 3 and rope_index.size(0) == 3:
                bsz, seqlen = input_ids.shape
                if attention_mask is not None:
                    text_pos_full = attention_mask.long().cumsum(-1) - 1
                    text_pos_full = torch.clamp(text_pos_full, min=0).to(
                        dtype=rope_index.dtype, device=rope_index.device
                    )
                    text_pos_full = text_pos_full.unsqueeze(0)  # (1, bsz, seqlen)
                else:
                    text_pos_full = (
                        torch.arange(seqlen, dtype=rope_index.dtype, device=rope_index.device)
                        .view(1, 1, -1)
                        .expand(1, bsz, -1)
                    )
                rope_index = torch.cat([text_pos_full, rope_index], dim=0)  # (4, bsz, seqlen)

            # (C, bsz, seqlen) -> (bsz, C, seqlen) to put it into DataProto,
            # transpose it back to (C, bsz, seqlen) before forward for model.
            rope_index = rope_index.transpose(0, 1)
            return {"position_ids": rope_index}

        return extra_data_provider

    def default_extra_data_provider(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen = input_ids.shape
        position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if attention_mask is not None:
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return {"position_ids": position_ids}

    return default_extra_data_provider


def find_vlm_text_decoder(model: nn.Module) -> Optional[nn.Module]:
    """
    Best-effort extractor for the text decoder stack of common VLM wrappers.
    """
    # Unwrap PEFT if present.
    base = getattr(model, "get_base_model", None)
    if callable(base):
        model = base()

    # Common attribute patterns across HF VLMs.
    for path in (
        ("language_model",),
        ("model", "language_model"),
        ("model", "text_model"),
        ("text_model",),
        ("model",),
    ):
        cur: Any = model
        ok = True
        for p in path:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and isinstance(cur, nn.Module):
            # Heuristic: the decoder usually has an embedding or layers attr.
            return cur
    return None


def get_vl_model_vision_tower_blocks(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance

    Reference: https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L128-L138
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual.blocks
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual.blocks
    return None


def _is_moe_config(cfg) -> bool:
    if cfg is None:
        return False
    # Heuristic: cover common HF config fields for MoE models.
    moe_keys = (
        "num_experts",
        "n_experts",
        "moe_num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "router_aux_loss_coef",
        "output_router_logits",
        "moe_layer_freq",
    )
    return any(getattr(cfg, k, None) not in (None, 0, False) for k in moe_keys)
