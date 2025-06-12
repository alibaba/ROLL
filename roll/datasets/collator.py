import json
import inspect
from collections import defaultdict

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Sequence, Union, Optional, List
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase, ProcessorMixin, BatchFeature
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
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


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
    tokenizer: PreTrainedTokenizerBase
    processor: ProcessorMixin
    extra_data_provider: Optional[callable] = None
    prompt_key: str = "prompt"
    answer_key: Optional[str] = "ground_truth"
    image_key: Optional[str] = "image"
    image_flag_key: Optional[str] = "image_flag"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # model_inputs for hf/deepspeed: input_id, attention_mask, pixel_values, image_grid_thw
        padded_features = defaultdict(list)
        un_padded_features = defaultdict(list)
        mm_feature_keys = set()
        for feature in features:
            # cannot process as batch directly though processor output as batch
            # since pixel_values would be packed among batch images while DataProto
            # requires all data fields has same batch size
            # if image is None, model_inputs would not inlcude image feature field
            model_inputs: BatchFeature = self.processor(
                images=feature[self.image_key] if self.image_key and feature[self.image_flag_key] else None,
                text=feature[self.prompt_key],
            )
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
                un_padded_features["multi_modal_data"].append(
                    {
                        "prompt_token_ids":  # different with input_ids
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                        "multi_modal_data": {"image": [feature[self.image_key]]},
                    }
                    if feature[self.image_flag_key]
                    else {
                        "prompt_token_ids":  # different with input_ids
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                    }
                )
            un_padded_features[self.answer_key].append(feature[self.answer_key])

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
                batch[key] = np.array(batch[key], dtype=object)
        return batch
