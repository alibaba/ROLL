# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Any

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput

from trl import PreTrainedModelWrapper


def value_head_load_state_dict(self: PreTrainedModelWrapper, state_dict: Dict[str, Any], strict=False) -> None:
    for name in list(state_dict.keys()):
        if name.startswith("v_head."):
            state_dict[name] = state_dict.pop(name)
        else:
            state_dict[name.replace("pretrained_model.", "")] = state_dict.pop(name)
    pretrained_model = getattr(self, "pretrained_model", None)
    if pretrained_model is not None:
        pretrained_model.load_state_dict(state_dict, strict=False)
        v_head: nn.Module = getattr(self, "v_head", None)
        if v_head is not None:
            for k in list(state_dict.keys()):
                if "v_head." in k:
                    state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
            v_head.load_state_dict(state_dict, strict=False)
    else:
        self.load_state_dict(state_dict, strict=False)


def token_classifier_forward(
    self: PreTrainedModelWrapper,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    return_past_key_values=False,
    **kwargs,
) -> TokenClassifierOutput:
    kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
    kwargs["past_key_values"] = past_key_values

    if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
        kwargs.pop("past_key_values")

    base_model_output = self.pretrained_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    last_hidden_state = base_model_output.hidden_states[-1]
    if last_hidden_state.device != self.v_head.summary.weight.device:
        last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

    value = self.v_head(last_hidden_state)

    return TokenClassifierOutput(
        loss=None,
        logits=value,
        hidden_states=base_model_output.hidden_states,
        attentions=base_model_output.attentions,
    )


def no_set_device_hook_post_init(self, state_dict):
    r"""
    We add the state dictionary of the value head to the state dictionary of the wrapped model
    by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
    keys of the value head state dictionary.
    """
    for k in list(state_dict.keys()):
        if "v_head." in k:
            state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
    self.v_head.load_state_dict(state_dict, strict=False)
    del state_dict

    if hasattr(self.pretrained_model, "hf_device_map"):
        if (
            "cpu" in self.pretrained_model.hf_device_map.values()
            or "disk" in self.pretrained_model.hf_device_map.values()
        ):
            raise ValueError(
                "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
            )

        # get the lm_head device
        for name, module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                lm_head_device = module.weight.device
                break

        # put v_head on the same device as the lm_head to avoid issues
        self.v_head = self.v_head.to(lm_head_device)

        def set_device_hook(module, input, outputs):
            r"""
            A hook that sets the device of the output of the model to the device of the first
            parameter of the model.

            Args:
                module (`nn.Module`):
                    The module to which the hook is attached.
                input (`tuple`):
                    The input to the module.
                outputs (`tuple`):
                    The output of the module.
            """
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        outputs[k] = v.to(lm_head_device)
                new_output = outputs
            else:
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
            return new_output

        self.register_forward_hook(set_device_hook)
        self.is_sequential_parallel = True
