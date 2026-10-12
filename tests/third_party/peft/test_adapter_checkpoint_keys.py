"""PEFT resume must preserve Qwen4 GR module names containing 'weight'."""
import copy

import pytest
import torch

peft = pytest.importorskip("peft")


@pytest.mark.parametrize("adapter_name", ["default", "domain"])
def test_restore_adapter_preserves_gr_module_names_and_forward(adapter_name):
    class Mixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_mix_weight_down = torch.nn.Linear(8, 4, bias=False)
            self.input_mix_weight_up = torch.nn.Linear(4, 8, bias=False)

        def forward(self, inputs):
            return self.input_mix_weight_up(self.input_mix_weight_down(inputs))

    torch.manual_seed(17)
    backbone = Mixer()
    config = peft.LoraConfig(r=2, lora_alpha=2, target_modules=[
        "input_mix_weight_down", "input_mix_weight_up"])
    source = peft.get_peft_model(copy.deepcopy(backbone), config, adapter_name=adapter_name)
    with torch.no_grad():
        for index, (name, parameter) in enumerate(source.named_parameters()):
            if "lora_" in name:
                parameter.copy_(torch.arange(parameter.numel()).view_as(parameter) * 0.03 + index)
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
    expected = source(inputs).detach()
    state = copy.deepcopy(peft.get_peft_model_state_dict(source, adapter_name=adapter_name))
    restored = peft.get_peft_model(copy.deepcopy(backbone), config, adapter_name=adapter_name)
    assert not torch.equal(restored(inputs), expected)
    result = peft.set_peft_model_state_dict(restored, state, adapter_name=adapter_name)
    assert not result.unexpected_keys
    assert not [key for key in result.missing_keys if "lora_" in key]
    torch.testing.assert_close(restored(inputs), expected, atol=0, rtol=0)
    torch.testing.assert_close(restored.state_dict(), source.state_dict(), atol=0, rtol=0)
