"""Public legacy adapter export must preserve Qwen4's vLLM 2D layout."""
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('RUN_QWEN4_STREAMING_TESTS') != '1',
                                reason='requires installed Megatron converter')


def _fixture(tmp_path, monkeypatch, adapter_names=('default',), broken=None):
    from peft import LoraConfig
    from megatron.core import mpu
    mpu.set_expert_tensor_parallel_rank(0)
    mpu.set_expert_tensor_parallel_world_size(1)
    from mcore_adapter.models.converter.post_converter import LoRAHFConverter
    from test_qwen4_exp_lora_conversion import _converter

    cfg = _converter().mca_config
    cfg.expert_model_parallel_size = 1
    cfg.num_moe_experts = 64
    cfg.moe_router_topk = 10
    cfg.params_dtype = torch.float32
    source = tmp_path / 'source'
    output = tmp_path / 'export'
    source.mkdir()
    # Native legacy producer layout and native PEFT config serializer.
    source_weights = {}
    for module, rank, inputs, outputs in [
        ('mlp.experts.linear_fc1', 6, 8, 32),
        ('mlp.experts.linear_fc2', 6, 16, 8),
        ('mlp.shared_experts.linear_fc1', 64, 8, 32),
        ('mlp.shared_experts.linear_fc2', 64, 16, 8),
        ('attn_hyper_connection.input_mix_weight_down', 64, 32, 4),
        ('attn_hyper_connection.input_mix_weight_up', 64, 4, 4),
        ('attn_hyper_connection.block_inject_weight', 64, 32, 4),
    ]:
        suffix = 'weight0' if '.experts.' in module else 'weight'
        prefix = 'decoder.layers.0.' + module
        source_weights[f'{prefix}.lora_A.{suffix}'] = (torch.arange(rank*inputs).reshape(rank, inputs).float()+1)/256
        source_weights[f'{prefix}.lora_B.{suffix}'] = (torch.arange(outputs*rank).reshape(outputs, rank).float()+2)/512
    if broken == 'missing_b':
        source_weights.pop('decoder.layers.0.mlp.experts.linear_fc2.lora_B.weight0')
    elif broken == 'bad_rank':
        name = 'decoder.layers.0.mlp.experts.linear_fc2.lora_B.weight0'
        source_weights[name] = source_weights[name][:, :5].clone()
    elif broken == 'oversized_rank':
        name = 'decoder.layers.0.mlp.experts.linear_fc2.lora_A.weight0'
        source_weights[name] = torch.ones(65, 16)
        source_weights['decoder.layers.0.mlp.experts.linear_fc2.lora_B.weight0'] = torch.ones(8, 65)
    for name in adapter_names:
        adapter = source / name
        LoraConfig(r=64, lora_alpha=128, target_modules=['linear_fc1', 'linear_fc2']).save_pretrained(adapter)
        payload = adapter / 'iter_0000001/mp_rank_00/model_optim_rng.pt'
        payload.parent.mkdir(parents=True)
        torch.save({'model':source_weights}, payload)
        (adapter/'latest_checkpointed_iteration.txt').write_text('1')
    # Isolate tokenizer/config copying; use real streaming, conversion and serialization.
    item = object.__new__(LoRAHFConverter)
    item.mca_config = cfg
    item.hf_config = type('Config', (), {'model_type':'qwen4_exp'})()
    item.checkpoint_path = str(source)
    item.adapter_name_or_path = str(source)
    item.save_directory = str(output)
    item.hf_base_model_path = str(tmp_path/'base-model-must-not-load')
    item.torch_dtype = torch.float32
    item.verbose = False
    def forbidden_base():
        raise AssertionError('Qwen4 2D adapter export must not materialize an HF base model')
    monkeypatch.setattr(item, '_get_hf_model_class', forbidden_base)
    def finalize():
        (output/'finalization-called').write_text('yes')
    monkeypatch.setattr(item, '_finalize', finalize)
    return item, source_weights, output


@pytest.mark.parametrize('names', [('default',), ('default','other'), ('other',)])
def test_public_export_preserves_routed_and_shared_experts(tmp_path, monkeypatch, names):
    from safetensors.torch import load_file
    from vllm.lora.peft_helper import PEFTHelper
    item, source, output = _fixture(tmp_path, monkeypatch, names)
    item.convert()
    assert (output/'finalization-called').is_file()
    for name in names:
        folder = output if name == 'default' else output/name
        saved = load_file(folder/'adapter_model.safetensors')
        config = json.loads((folder/'adapter_config.json').read_text())
        assert config['r'] == 64 and config['lora_alpha'] == 128
        assert config.get('rank_pattern', {}) == {}
        assert config['roll_lora_layout'] == 'qwen4_exp_vllm_2d'
        helper = PEFTHelper.from_dict(config)
        assert helper.vllm_lora_scaling_factor == 2
        expected = {}
        for family, rank in [('experts.0',6), ('shared_expert',64)]:
            mca = 'experts' if family == 'experts.0' else 'shared_experts'
            suffix = 'weight0' if mca == 'experts' else 'weight'
            for projection, fc, half in [('gate_proj','linear_fc1',0), ('up_proj','linear_fc1',1), ('down_proj','linear_fc2',None)]:
                a = source[f'decoder.layers.0.mlp.{mca}.{fc}.lora_A.{suffix}']
                b = source[f'decoder.layers.0.mlp.{mca}.{fc}.lora_B.{suffix}']
                if half is not None:
                    b = b.chunk(2, dim=0)[half]
                prefix = f'model.language_model.layers.0.mlp.{family}.{projection}'
                expected[prefix+'.lora_A.weight'] = a
                expected[prefix+'.lora_B.weight'] = b
                got_a, got_b = saved[prefix+'.lora_A.weight'], saved[prefix+'.lora_B.weight']
                assert got_a.shape[0] == got_b.shape[1] == rank
                torch.testing.assert_close(got_b @ got_a * helper.vllm_lora_scaling_factor, b @ a * 2, rtol=0,atol=0)
        for projection in ['input_mix_weight_down','input_mix_weight_up','block_inject_weight']:
            for factor in ['A','B']:
                expected[f'model.language_model.layers.0.attn_hyper_connection.{projection}.lora_{factor}.weight'] = source[f'decoder.layers.0.attn_hyper_connection.{projection}.lora_{factor}.weight']
        assert saved.keys() == expected.keys()
        for key, tensor in expected.items():
            torch.testing.assert_close(saved[key],tensor,rtol=0,atol=0)


@pytest.mark.parametrize('broken', ['missing_b','bad_rank','oversized_rank'])
def test_public_export_rejects_invalid_adapter_pairs(tmp_path, monkeypatch, broken):
    item, _, output = _fixture(tmp_path, monkeypatch, broken=broken)
    with pytest.raises(ValueError, match='LoRA|adapter|rank'):
        item.convert()
    assert not list(output.rglob('adapter_model.safetensors'))


@pytest.mark.parametrize('unsupported', [
    {'rank_pattern': {'down_proj': 6}},
    {'alpha_pattern': {'down_proj': 12}},
    {'bias': 'all'},
    {'use_dora': True},
    {'use_rslora': True},
    {'modules_to_save': ['lm_head']},
])
def test_public_export_rejects_unsupported_modes_before_writing(tmp_path, monkeypatch, unsupported):
    item, _, output = _fixture(tmp_path, monkeypatch, ('default', 'other'))
    config_path = Path(item.adapter_name_or_path) / 'other/adapter_config.json'
    config = json.loads(config_path.read_text())
    config.update(unsupported)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='uniform scaling'):
        item.convert()
    # A valid first adapter must not be published when a later one is invalid.
    assert not list(output.rglob('adapter_model.safetensors'))
    assert not (output / 'finalization-called').exists()


@pytest.mark.parametrize('corruption,reason', [
    ('conflicting_duplicate', 'Conflicting replicated LoRA tensor'),
    ('empty', 'Empty LoRA adapter'),
    ('invalid_name', 'Invalid Qwen4 LoRA tensor'),
    ('invalid_dimensions', 'Invalid Qwen4 LoRA tensor'),
])
def test_public_export_rejects_corrupt_converted_stream(tmp_path, monkeypatch, corruption, reason):
    item, _, output = _fixture(tmp_path, monkeypatch)
    native_stream = item._stream_hf_weights

    def corrupt_stream(*args, **kwargs):
        # Run the actual legacy reader and conversion, then corrupt its output
        # at the writer boundary whose contract is under test.
        converted = list(native_stream(*args, **kwargs))
        assert converted
        if corruption == 'empty':
            return
        name, weight = converted[0]
        if corruption == 'invalid_name':
            converted[0] = ('unexpected.weight', weight)
        elif corruption == 'invalid_dimensions':
            converted[0] = (name, weight.unsqueeze(0))
        yield from converted
        if corruption == 'conflicting_duplicate':
            yield name, weight + 1

    monkeypatch.setattr(item, '_stream_hf_weights', corrupt_stream)
    with pytest.raises(ValueError, match=reason):
        item.convert()
    assert not list(output.rglob('adapter_model.safetensors'))
    assert not (output / 'finalization-called').exists()
