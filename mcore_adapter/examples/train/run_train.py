import functools
import hashlib
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import torch
from filelock import FileLock
from huggingface_hub import snapshot_download
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.collator import PairwiseDataCollatorWithPadding, SFTDataCollatorWith4DAttentionMask
from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from llamafactory.model import load_tokenizer
from llamafactory.train.callbacks import SaveProcessorCallback
from llamafactory.train.dpo import run_dpo
from llamafactory.train.pt import run_pt
from llamafactory.train.sft import run_sft
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq, HfArgumentParser
from transformers.trainer_callback import TrainerCallback

from mcore_adapter.adapters import apply_megatron_lora, find_all_linear_modules, set_linear_is_expert
from mcore_adapter.models import AutoConfig, AutoModel
from mcore_adapter.trainer import DPOTrainer, McaTrainer
from mcore_adapter.trainer.dpo_config import DPOConfig
from mcore_adapter.training_args import Seq2SeqTrainingArguments


def download_model(model_name_or_path: str, local_dir: str = None) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    use_model_scope = os.getenv("USE_MODELSCOPE", "0") == "1"
    temp_lock_path = os.path.join(
        "~/.cache/mcore_adapter/temp_lock",
        f"{hashlib.md5(model_name_or_path.encode()).hexdigest()}.lock",
    )
    with FileLock(temp_lock_path):
        if use_model_scope:
            from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

            return ms_snapshot_download(model_name_or_path, local_dir=local_dir)
        return snapshot_download(model_name_or_path, local_dir=local_dir)


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


@dataclass
class UseMcaArguments:
    use_mca: bool = field(default=True)


def get_args() -> Tuple[
    Seq2SeqTrainingArguments,
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    UseMcaArguments,
]:
    training_args, model_args, data_args, finetuning_args, generating_args, use_mca_args = HfArgumentParser(
        (
            Seq2SeqTrainingArguments,
            ModelArguments,
            DataArguments,
            FinetuningArguments,
            GeneratingArguments,
            UseMcaArguments,
        )
    ).parse_args_into_dataclasses()
    if not use_mca_args.use_mca:
        from transformers import Seq2SeqTrainingArguments as HFSeq2SeqTrainingArguments

        training_args, model_args, data_args, finetuning_args, generating_args, use_mca_args = HfArgumentParser(
            (
                HFSeq2SeqTrainingArguments,
                ModelArguments,
                DataArguments,
                FinetuningArguments,
                GeneratingArguments,
                UseMcaArguments,
            )
        ).parse_args_into_dataclasses()
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)
    return training_args, model_args, data_args, finetuning_args, generating_args, use_mca_args


def data_collator_wrapper(data_collator):
    @functools.wraps(data_collator)
    def wrapper(features: Sequence[Dict[str, Any]]):
        labels_key = [k for k in features[0].keys() if k.endswith("labels")]
        input_ids_key = [k for k in features[0].keys() if k.endswith("input_ids")]
        for feature in features:
            if len(labels_key) == 0:  # pt
                feature["labels"] = deepcopy(feature["input_ids"])[1:]
            for k in labels_key:
                feature[k] = feature[k][1:]
            for k in input_ids_key:
                feature[k] = feature[k][:-1]
            for k in ["attention_mask", "position_ids"]:
                if k in feature:
                    feature[k] = feature[k][:-1]
        return data_collator(features)

    return wrapper


def setup_lora_training(model, finetuning_args):
    model.enable_input_require_grads()
    target_modules = find_all_linear_modules(model)

    peft_kwargs = {
        "r": finetuning_args.lora_rank,
        "target_modules": target_modules,
        "lora_alpha": finetuning_args.lora_alpha,
        "lora_dropout": finetuning_args.lora_dropout,
    }

    lora_config = LoraConfig(
        **peft_kwargs,
    )
    model = get_peft_model(model, lora_config)

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    return model


def pt_mca_train(
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    finetuning_args: FinetuningArguments,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
    if finetuning_args.finetuning_type == "lora":
        apply_megatron_lora()
        set_linear_is_expert(model[0])
        model.models[0] = setup_lora_training(model[0], finetuning_args)
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    data_args.cutoff_len -= 1
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    data_collator = data_collator_wrapper(data_collator)
    trainer = McaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **dataset_module,
    )
    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
    #                                     torch.profiler.ProfilerActivity.CUDA],
    #                         schedule=torch.profiler.schedule(skip_first=0, wait=0, warmup=1, active=2, repeat=1),
    #                         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"output_dir_tp2pp1_{training_args.process_index}"),
    #                         profile_memory=True,
    #                         with_stack=True,
    #                         record_shapes=True) as prof:
    #     trainer.add_callback(ProfCallback(prof=prof))
    #     trainer.train()
    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))
    trainer.train(training_args.resume_from_checkpoint)


def sft_mca_train(
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    finetuning_args: FinetuningArguments,
):
    data_args.neat_packing = training_args.sequence_packing = data_args.neat_packing or training_args.sequence_packing
    data_args.packing = data_args.neat_packing or data_args.packing
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    data_args.cutoff_len -= 1
    if finetuning_args.finetuning_type == "lora":
        apply_megatron_lora()
        set_linear_is_expert(model[0])
        model.models[0] = setup_lora_training(model[0], finetuning_args)
    if model.config.hf_model_type in ["qwen2_vl"] and finetuning_args.freeze_vision_tower:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ["vision_model.blocks", "vision_model.patch_embed"]):
                p.requires_grad_(False)
    if model.config.hf_model_type in ["qwen2_vl"] and finetuning_args.freeze_multi_modal_projector:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ["multi_modal_projector"]):
                p.requires_grad_(False)
    if model.config.hf_model_type in ["qwen2_vl"] and finetuning_args.freeze_language_model:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ["embedding", "decoder", "output_layer"]):
                p.requires_grad_(False)
    pad_to_max = training_args.expert_model_parallel_size is not None and training_args.expert_model_parallel_size > 1 and not training_args.variable_seq_lengths
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        padding="max_length" if pad_to_max else "longest",
        max_length=data_args.cutoff_len if pad_to_max else None,
        pad_to_multiple_of=64,
        label_pad_token_id=-100,
        **tokenizer_module,
    )
    data_collator = data_collator_wrapper(data_collator)
    trainer = McaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **dataset_module,
    )
    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))
    trainer.train(training_args.resume_from_checkpoint)


def dpo_mca_train(
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    finetuning_args: FinetuningArguments,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
    if finetuning_args.use_ref_model:
        ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path, training_args)
        ref_model = AutoModel.from_config(ref_config)
        ref_model.load_state_dict(model.state_dict())
    else:
        ref_model = None
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    data_args.cutoff_len -= 1
    pad_to_max = training_args.expert_model_parallel_size is not None and training_args.expert_model_parallel_size > 1
    dpo_config = DPOConfig(
        beta=finetuning_args.pref_beta,
        pref_loss=finetuning_args.pref_loss,
        label_smoothing=finetuning_args.dpo_label_smoothing,
    )
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=64,
        padding="max_length" if pad_to_max else "longest",
        max_length=data_args.cutoff_len if pad_to_max else None,
        label_pad_token_id=-100,
        **tokenizer_module,
    )
    data_collator = data_collator_wrapper(data_collator)
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_config=dpo_config,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **dataset_module,
    )
    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))
    trainer.train(training_args.resume_from_checkpoint)


def mca_train(
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    finetuning_args: FinetuningArguments,
):
    if finetuning_args.stage == "pt":
        pt_mca_train(training_args, model_args, data_args, finetuning_args)
    elif finetuning_args.stage == "sft":
        sft_mca_train(training_args, model_args, data_args, finetuning_args)
    elif finetuning_args.stage == "dpo":
        dpo_mca_train(training_args, model_args, data_args, finetuning_args)
    else:
        raise ValueError("Unknown task: {}.".format(finetuning_args.stage))


def llama_factory_train(training_args, model_args, data_args, finetuning_args, generating_args):
    data_args.cutoff_len += 1
    callbacks = None
    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task: {}.".format(finetuning_args.stage))


def main():
    training_args, model_args, data_args, finetuning_args, generating_args, use_mca_args = get_args()
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    if use_mca_args.use_mca:
        mca_train(training_args, model_args, data_args, finetuning_args)
    else:
        llama_factory_train(training_args, model_args, data_args, finetuning_args, generating_args)


if __name__ == "__main__":
    main()
