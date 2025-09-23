import os
from typing import List, Optional

import torch
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


try:
    from mcore_adapter import TrainingArguments as mca_TrainingArguments
    from mcore_adapter.models import AutoModel
except Exception as e:
    mca_TrainingArguments = None

from roll.configs import ModelArguments
from roll.utils.checkpoint_manager import download_model, file_lock_context
from roll.utils.logging import get_logger
from roll.utils.packages import is_transformers_version_greater_than


logger = get_logger()


def prepare_automap_files(model_path: str):
    python_files = []
    for file_name in os.listdir(model_path):
        if file_name.endswith(".py") and os.path.isfile(os.path.join(model_path, file_name)):
            python_files.append(file_name)
    with file_lock_context(model_path):
        for file_name in python_files:
            get_cached_module_file(model_path, file_name)


def default_tokenizer_provider(model_args: "ModelArguments", model_name_or_path: str=None):
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


def default_processor_provider(model_args: "ModelArguments", model_name_or_path: str=None):
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


def setup_lora_training(config, model, model_args: "ModelArguments", is_trainable: Optional[bool] = False):
    model.enable_input_require_grads()
    if is_trainable:
        target_modules = model_args.lora_target
        lora_config = {
            "task_type": TaskType.CAUSAL_LM,
            "r": model_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
            "modules_to_save": model_args.additional_target,
        }
        model = get_peft_model(model, LoraConfig(**lora_config))
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
    init_kwargs = {"trust_remote_code": True, **model_args.model_config_kwargs}
    config = AutoConfig.from_pretrained(model_name_or_path, **init_kwargs)
    if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
        setattr(config, "_attn_implementation", model_args.attn_implementation)
    if not is_trainable:
        setattr(config, "use_cache", True)
    else:
        setattr(config, "use_cache", False)
    if model_args.moe_aux_loss_coef is not None:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)
        setattr(config, "output_router_logits", is_trainable)
    init_kwargs["low_cpu_mem_usage"] = not is_deepspeed_zero3_enabled()
    if not is_deepspeed_zero3_enabled() and not is_fsdp_enabled():
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
    model = model_class.from_pretrained(**init_kwargs)
    if not model_args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    return model


def patch_model(model, config, use_mcore):
    import types

    model_type = config.model_type

    forward_patch = None
    # patch to force vit forward with mock image to avoid hang
    if use_mcore:
        if "qwen2_vl" == model_type or "qwen2_5_vl" == model_type:

            def forward_patch(
                self,
                input_ids: "torch.Tensor",
                position_ids: Optional["torch.Tensor"] = None,
                attention_mask: Optional["torch.Tensor"] = None,
                decoder_input: Optional["torch.Tensor"] = None,
                labels: Optional["torch.Tensor"] = None,
                pixel_values: Optional["torch.Tensor"] = None,
                pixel_values_videos: Optional["torch.Tensor"] = None,
                image_grid_thw: Optional["torch.LongTensor"] = None,
                video_grid_thw: Optional["torch.LongTensor"] = None,
                second_per_grid_ts: Optional[torch.Tensor] = None,  # for videos
                **kwargs,
            ):
                force_vit_image = kwargs.pop("force_vit_image", False)
                force_vit_video = kwargs.pop("force_vit_video", False)
                if position_ids is None and input_ids is not None:
                    position_ids, _ = self.get_rope_index(
                        input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask
                    )
                cp_batch = {
                    "position_ids": position_ids,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if self.config.context_parallel_size > 1:
                    cp_batch = {k: v.clone() if v is not None else None for k, v in cp_batch.items()}
                    cp_batch = (
                        type(self)
                        .mro()[1]
                        .get_batch_on_this_cp_rank(self, cp_batch, dim3_keys=["attention_mask", "position_ids"])
                    )
                if (
                    not self.pre_process
                    or (pixel_values is None and pixel_values_videos is None)
                    or decoder_input is not None
                ):
                    return (
                        type(self)
                        .mro()[1]
                        .forward(self, decoder_input=decoder_input, labels=labels, **cp_batch, **kwargs)
                    )
                inputs_ranges = self.get_input_ranges(input_ids.shape[1])
                inputs_embeds = self.embedding(input_ids=cp_batch["input_ids"], position_ids=cp_batch["position_ids"])
                if pixel_values is not None:
                    inputs_embeds = self.construct_inputs_embeds(
                        input_ids,
                        inputs_embeds,
                        pixel_values,
                        image_grid_thw,
                        inputs_ranges,
                        self.config.image_token_id,
                    )
                elif force_vit_image:
                    # force vit forward with mock image to avoid hang
                    inputs_embeds = self._handle_missing_visual(inputs_embeds)
                if pixel_values_videos is not None:
                    inputs_embeds = self.construct_inputs_embeds(
                        input_ids,
                        inputs_embeds,
                        pixel_values_videos,
                        video_grid_thw,
                        inputs_ranges,
                        self.config.video_token_id,
                    )
                elif force_vit_video:
                    # force vit forward with mock image to avoid hang
                    inputs_embeds = self._handle_missing_visual(inputs_embeds)
                decoder_input = inputs_embeds
                return (
                    type(self).mro()[1].forward(self, decoder_input=decoder_input, labels=labels, **cp_batch, **kwargs)
                )

        if forward_patch is not None:
            for model_chunk in model.get_models():
                model_chunk.forward = types.MethodType(forward_patch, model_chunk)
    else:
        if "qwen2_vl" == model_type or "qwen2_5_vl" == model_type:
            if is_peft_model := (getattr(model, "peft_config", None) is not None):
                ori_forward = type(model.get_base_model()).forward
            else:
                ori_forward = type(model).forward

            def _handle_missing_visual(self, inputs_embeds: "torch.FloatTensor"):
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
                    inputs_embeds = self.model.embed_tokens(input_ids)
                    inputs_embeds = _handle_missing_visual(self, inputs_embeds)
                if kwargs.pop("force_vit_video", False) and pixel_values_videos is None:
                    if inputs_embeds is None:
                        inputs_embeds = self.model.embed_tokens(input_ids)
                    # force vit forward with mock image to avoid hang
                    inputs_embeds = _handle_missing_visual(self, inputs_embeds)
                return ori_forward(
                    self,
                    input_ids,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                    pixel_values,
                    pixel_values_videos,
                    image_grid_thw,
                    video_grid_thw,
                    rope_deltas,
                    cache_position,
                )

        if forward_patch is not None:
            if is_peft_model:
                model.get_base_model().forward = types.MethodType(forward_patch, model.get_base_model())
            else:
                model.forward = types.MethodType(forward_patch, model)


def default_diffusion_module_provider(
    tokenizer: None,
    model_args: ModelArguments,
    training_args: TrainingArguments = None,
    is_trainable: Optional[bool] = False,
):
    if model_args.model_config_kwargs["model_name"] == "wan2_2":
        from roll.pipeline.diffusion.modules.wan_module import WanTrainingModule
        print(f"{model_args.model_config_kwargs=}")
        training_module =  WanTrainingModule(**model_args.model_config_kwargs)
    else:
        raise NotImplementedError(f"model_type {model_args.model_type} not implemented yet")

    return training_module


def default_actor_model_provider(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    training_args: "TrainingArguments" = None,
    is_trainable: Optional[bool] = False,
):
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    old_model_name_or_path = model_args.model_name_or_path
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)
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
        freeze_model(model, model_args)
        patch_model(model, config, use_mcore=True)
    else:
        # hf
        init_kwargs = {
            "torch_dtype": model_args.compute_dtype,
            "trust_remote_code": True,
        }
        if not is_deepspeed_zero3_enabled():
            init_kwargs["low_cpu_mem_usage"] = True
            if is_trainable:
                init_kwargs["device_map"] = {"": torch.cuda.current_device()}
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
                init_kwargs["device_map"] = {"": torch.cuda.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
        logger.info(f"init_kwargs: {init_kwargs}")
        if model_args.model_type in ["auto_sequence_classification"]:
            logger.info(f"use AutoModelForSequenceClassification model {model_args.model_type}")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.num_labels = model_args.num_labels
            if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
                setattr(config, "_attn_implementation", model_args.attn_implementation)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path, config=config, **init_kwargs
            )
        elif model_args.model_type in ["auto_token_classification"]:
            logger.info(f"use AutoModelForTokenClassification model {model_args.model_type}")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.num_labels = model_args.num_labels
            if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
                setattr(config, "_attn_implementation", model_args.attn_implementation)
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
            setattr(model, "load_state_dict", value_head_load_state_dict.__get__(model))
            logger.info(f"patch AutoModelForCausalLMWithValueHead load_state_dict and forward")
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
                init_kwargs["device_map"] = {"": torch.cuda.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
        logger.info(f"init_kwargs: {init_kwargs}")
        if model_args.model_type in ["auto_token_classification"]:
            logger.info(f"use AutoModelForTokenClassification model {model_args.model_type}")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.num_labels = model_args.num_labels
            if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
                setattr(config, "_attn_implementation", model_args.attn_implementation)
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
            setattr(model, "load_state_dict", value_head_load_state_dict.__get__(model))
        else:
            raise NotImplementedError
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model_args.model_name_or_path = old_model_name_or_path

    return model


def get_extra_data_provider(model_name_or_path: str, processor=None):
    model_name_or_path = download_model(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    if "qwen2" in config.model_type:
        import types

        from transformers import BatchFeature  # help define a object to accesss attr

        dummy_self = BatchFeature(
            {
                "config": BatchFeature(
                    {
                        "vision_config": BatchFeature({"spatial_merge_size": processor.image_processor.merge_size}),
                        "image_token_id": processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                        "video_token_id": processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                        "vision_start_token_id": processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                    }
                )
            }
        )
        if is_transformers_version_greater_than("4.52.0"):
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
        ):
            rope_index = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)[0]
            # (3, bsz, seqlen) -> (bsz, 3, seqlen) to put it into DataProto,
            # transpose it batck to (3, bsz, seqlen) before forward for model
            rope_index = rope_index.transpose(0, 1)
            return {"position_ids": rope_index}

        return extra_data_provider
    return None