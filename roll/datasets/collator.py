import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import BatchFeature, PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy


def collate_fn_to_dict_list(data_list: list[dict]) -> dict:
    """将list[dict]数据转成dict[list]"""
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.cat(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.empty(len(val), dtype=object)
        non_tensors[key][:] = val

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


@dataclass
class DataCollatorWithPaddingForDPO:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def pad_sequences(self, sequences: List[List[int]], pad_value: int = 0) -> torch.Tensor:
        padded = [seq + [pad_value] * (self.max_length - len(seq)) for seq in sequences]
        return torch.tensor(padded)

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        origin_batch_size = len(prompt_id_lens)
        input_ids = torch.stack((chosen_ids, reject_ids), dim=1).view(2 * origin_batch_size, -1)
        att_masks = torch.stack((c_mask, r_mask), dim=1).view(2 * origin_batch_size, -1)
        prompt_id_lens = torch.stack((prompt_id_lens, prompt_id_lens), dim=1).view(2 * origin_batch_size)
        return input_ids, att_masks, prompt_id_lens

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen_ids = []
        c_mask = []
        reject_ids = []
        r_mask = []
        prompt_ids_lens = []

        for item in batch:
            chosen_ids.append(item["chosen_ids"])
            c_mask.append(item["c_mask"])
            reject_ids.append(item["reject_ids"])
            r_mask.append(item["r_mask"])
            prompt_ids_lens.append(item["prompt_ids_lens"])

        chosen_ids = self.pad_sequences(chosen_ids, pad_value=self.tokenizer.pad_token_id)
        c_mask = self.pad_sequences(c_mask)
        reject_ids = self.pad_sequences(reject_ids, pad_value=self.tokenizer.pad_token_id)
        r_mask = self.pad_sequences(r_mask)
        prompt_ids_lens = torch.tensor(prompt_ids_lens)

        input_ids, attention_mask, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_ids_lens
        )
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "prompt_id_lens": prompt_id_lens, "position_ids": position_ids}


@dataclass
class DataCollatorWithPaddingForPaddedKeys:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_features = [{k: v for k, v in feature.items() if k in self.padded_keys} for feature in features]
        un_padded_features = [{k: v for k, v in feature.items() if k not in self.padded_keys} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["position_ids"] = torch.clip(torch.cumsum(batch["attention_mask"], dim=-1) - 1, min=0, max=None)
        un_padded_batch = collate_fn_to_dict_list(un_padded_features)
        batch.update(un_padded_batch)
        return batch


@dataclass
class DataCollatorWithPaddingForMM:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    processor: Optional[ProcessorMixin] = None
    extra_data_provider: Optional[callable] = None
    prompt_key: str = "prompt"
    answer_key: Optional[str] = "ground_truth"
    image_key: Optional[str] = "image"
    image_flag_key: Optional[str] = "image_flag"
    video_key: Optional[str] = "video"
    video_flag_key: Optional[str] = "video_flag"
    image_placeholder: Optional[str] = None
    image_token: str = "<|vision_start|><|image_pad|><|vision_end|>"
    video_placeholder: Optional[str] = None
    video_token: str = "<|vision_start|><|video_pad|><|vision_end|>"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])
    extra_unpadded_keys: List[str] = field(default_factory=lambda: [])
    return_tensors: str = "pt"
    return_infer_inputs: bool = True  # whether to include infer engine inputs which differs with train

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.tokenizer and self.processor
        # model_inputs for hf/deepspeed: input_id, attention_mask, pixel_values, image_grid_thw
        padded_features = defaultdict(list)
        un_padded_features = defaultdict(list)
        mm_feature_keys = set()
        for feature in features:
            # cannot process as batch directly though processor output as batch
            # since pixel_values would be packed among batch images while DataProto
            # requires all data fields has same batch size
            # if image is None, model_inputs would not include image feature field
            prompt = feature[self.prompt_key]
            if not isinstance(prompt, str):
                prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if self.image_placeholder:
                prompt = prompt.replace(self.image_placeholder, self.image_token)
            if self.video_placeholder:
                prompt = prompt.replace(self.video_placeholder, self.video_token)
            # TODO: support video
            model_inputs: BatchFeature = self.processor(
                images=feature[self.image_key]
                if self.image_key and (not self.image_flag_key or feature[self.image_flag_key])
                else None,
                text=prompt,
            )
            if not isinstance(model_inputs, BatchFeature):
                model_inputs = BatchFeature(data=model_inputs)
            for key in ["prompt", "position_ids", "rope_deltas"]:  # remove unnecessary feature
                if key in model_inputs:
                    model_inputs.pop(key)
            for key in filter(lambda k: k in model_inputs, self.padded_keys):
                padded_features[key].append(model_inputs.pop(key)[0])
            # mm feature fileds can be different because of mixed data
            mm_feature_keys = mm_feature_keys.union(model_inputs.keys())
            # to tensors except padded_keys which would be converted after padding
            model_inputs.convert_to_tensors(tensor_type=self.return_tensors)
            if self.image_key:
                # allow mixed text and multi-modal data
                # assert model_inputs, "should have multi-modal features"
                # tensors in multi_modal_inputs dict have bsz=1 and should be
                # concat at dim=0 before model forward
                un_padded_features["multi_modal_inputs"].append(dict(model_inputs))
                # inputs for infer engine, not tensors
                if self.return_infer_inputs:
                    un_padded_features["multi_modal_data"].append(
                        {
                            "prompt_token_ids":  # different with input_ids
                            self.tokenizer.encode(prompt, add_special_tokens=False),
                            "multi_modal_data": {
                                "image": [feature[self.image_key]]
                                if not isinstance(feature[self.image_key], list)
                                else feature[self.image_key]
                            },
                        }
                        if (not self.image_flag_key or feature[self.image_flag_key]) and feature[self.image_key]
                        else {
                            "prompt_token_ids":  # different with input_ids
                            self.tokenizer.encode(prompt, add_special_tokens=False),
                        }
                    )
            if self.answer_key:
                un_padded_features[self.answer_key].append(feature[self.answer_key])
            if self.extra_unpadded_keys:
                for key in self.extra_unpadded_keys:
                    un_padded_features[key].append(feature[key])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update(un_padded_features)

        # other custom data fields: mainly for specific position_ids currently
        # position_ids for qwen2-vl is optional and make sure it is a 3D tensor
        # shaped with `(3, bs, seq_len)` for 3D-RoPE if provided, while we use
        # `(bs, 3, seq_len)` to put it into DataProto which limits batch size dim
        if self.extra_data_provider:
            fun_params = inspect.signature(self.extra_data_provider).parameters
            kwargs = {}
            for key in fun_params:
                if key in batch:
                    kwargs[key] = batch[key]
                elif key in mm_feature_keys:
                    mm_inputs = [inputs[key] for inputs in batch["multi_modal_inputs"] if key in inputs]
                    kwargs[key] = torch.concat(mm_inputs, dim=0) if mm_inputs else fun_params[key].default
                else:
                    kwargs[key] = fun_params[key].default
            extra_data = self.extra_data_provider(**kwargs)
            batch.update(extra_data)

        # each field should be a tensor or np.array(val=list_data, dtype=object)
        # to be stored in DataProto
        for key in batch:
            if isinstance(batch[key], (torch.Tensor, np.ndarray)):
                assert batch[key].shape[0] == batch["input_ids"].shape[0]
            else:
                assert len(batch[key]) == batch["input_ids"].shape[0]
                val = batch[key]
                batch[key] = np.empty(len(batch[key]), dtype=object)
                batch[key][:] = val
        return batch

@dataclass
class DataCollatorWithPaddingForMMWithLabels(DataCollatorWithPaddingForMM):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorForSFT(DataCollatorWithPaddingForPaddedKeys):
    label_pad_token_id: int = -100
    shift_feature: bool = True

    def __call__(self, features):
        padded_batch = super().__call__(features)
        labels = padded_batch.pop("labels")
        padded_labels = []
        for label in labels:
            seq_len = len(label)
            if seq_len > self.max_length:
                padded_labels.append(label[:self.max_length])
            else:
                padded_labels.append(label + [self.label_pad_token_id] * (self.max_length - seq_len))
        
        padded_batch.update({"labels": torch.tensor(padded_labels, dtype=torch.int64)})

        if self.shift_feature:
            labels = padded_batch.pop("labels")
            labels = labels[:, 1:]
            labels = torch.cat([labels, torch.tensor([self.label_pad_token_id] * labels.shape[0], dtype=torch.int64).reshape(-1, 1)], dim=1)
            padded_batch["labels"] = labels

        return padded_batch
