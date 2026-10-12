"""Exercise ROLL LoRA calls against explicit native API signatures without an engine."""
import ast
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_utils(monkeypatch, version):
    class Request:
        def __init__(self, **values):
            self.__dict__.update(values)

    class PeftHelper:
        @classmethod
        def from_dict(cls, config):
            return cls()

        @classmethod
        def from_local_dir(cls, path, max_position_embeddings, tensorizer_config_dict=None):
            helper = cls()
            helper.tensorizer = tensorizer_config_dict
            return helper

        def validate_legal(self, config):
            pass

    manager_class = type('Manager', (), {})
    _module(monkeypatch, 'vllm', __version__=version)
    _module(monkeypatch, 'vllm.lora.request', LoRARequest=Request)
    _module(monkeypatch, 'vllm.lora.utils', get_adapter_absolute_path=lambda path: path)
    _module(monkeypatch, 'vllm.lora.worker_manager', LRUCacheWorkerLoRAManager=manager_class)
    _module(monkeypatch, 'vllm.lora.peft_helper', PEFTHelper=PeftHelper)
    _module(monkeypatch, 'roll.third_party.vllm.compat',
            import_attribute=lambda modules, name: object, patch_moe_model_weight_loaders=lambda model: None)
    spec = importlib.util.spec_from_file_location('vllm_utils_under_test', ROOT/'roll/third_party/vllm/vllm_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_vllm_lora_manager()
    return module, manager_class, Request


def _manager(model_class, legacy=False):
    mapper = SimpleNamespace(get_unstacked_mapper=lambda: 'unstacked-mapper')
    config = SimpleNamespace(lora_dtype='bf16')
    if legacy:
        config.lora_extra_vocab_size = 16
    return SimpleNamespace(
        _adapter_manager=SimpleNamespace(supported_lora_modules=['qkv'],
            packed_modules_mapping={'qkv':['q_proj','k_proj','v_proj']},
            model=SimpleNamespace(hf_to_vllm_mapper=mapper, lora_skip_prefixes=['mtp.']),
            moe_ep_load_spec='expert-layout'),
        _lora_model_cls=model_class, lora_config=config, vocab_size=128,
        max_position_embeddings=2048, embedding_modules={'embed':'input_embeddings'},
        embedding_padding_modules=['embed'])


@pytest.mark.parametrize('version', ['0.1.dev20073+g8e685d198', '0.12.0'])
@pytest.mark.parametrize('from_tensors', [True, False])
def test_modern_loaders_use_native_arguments_and_model_mapping(monkeypatch, version, from_tensors):
    module, Manager, Request = _load_utils(monkeypatch, version)
    seen = {}

    class Model:
        @classmethod
        def from_lora_tensors(cls, lora_model_id, tensors, peft_helper, device='cuda', dtype=None,
                              model_vocab_size=None, weights_mapper=None, skip_prefixes=None):
            seen.update(locals())
            return SimpleNamespace()

        @classmethod
        def from_local_checkpoint(cls, path, expected_lora_modules, peft_helper, lora_model_id,
                                  device='cuda', dtype=None, model_vocab_size=None, weights_mapper=None,
                                  tensorizer_config_dict=None, skip_prefixes=None, moe_ep_spec=None):
            seen.update(locals())
            return SimpleNamespace()

    request_class = module.TensorLoRARequest if from_tensors else Request
    request = request_class(lora_int_id=7, lora_path='adapter', tensorizer_config_dict={'uri':'fixture'},
                            peft_config={}, lora_tensors={'q_proj': 'tensor'}, is_3d_lora_weight=True)
    output = Manager._load_adapter(_manager(Model), request)
    assert seen['model_vocab_size'] == 128
    assert seen['device'] == 'cpu' and seen['dtype'] == 'bf16'
    assert seen['weights_mapper'] == 'unstacked-mapper'
    assert seen['skip_prefixes'] == ['mtp.']
    assert output.is_3d_lora_weight is True
    if from_tensors:
        assert seen['tensors'] is request.lora_tensors
    else:
        assert seen['tensorizer_config_dict'] == {'uri':'fixture'}
        assert seen['peft_helper'].tensorizer == {'uri':'fixture'}
        assert seen['moe_ep_spec'] == 'expert-layout'
        assert set(seen['expected_lora_modules']) == {'q_proj','k_proj','v_proj'}


@pytest.mark.parametrize('from_tensors', [True, False])
def test_legacy_loaders_keep_embedding_arguments(monkeypatch, from_tensors):
    module, Manager, Request = _load_utils(monkeypatch, '0.8.4')
    seen = {}

    class Model:
        @classmethod
        def from_lora_tensors(cls, lora_model_id, tensors, peft_helper, device, dtype,
                              weights_mapper, embeddings, target_embedding_padding,
                              embedding_modules, embedding_padding_modules):
            seen.update(locals())
            return SimpleNamespace()

        @classmethod
        def from_local_checkpoint(cls, path, expected_lora_modules, peft_helper, lora_model_id,
                                  device, dtype, weights_mapper, target_embedding_padding,
                                  embedding_modules, embedding_padding_modules):
            seen.update(locals())
            return SimpleNamespace()

    request_class = module.TensorLoRARequest if from_tensors else Request
    request = request_class(lora_int_id=7, lora_path='adapter', peft_config={}, lora_tensors={})
    manager = _manager(Model, legacy=True)
    # Earlier models expose a mapper without unstacking support.
    manager._adapter_manager.model.hf_to_vllm_mapper = 'legacy-mapper'
    Manager._load_adapter(manager, request)
    assert seen['target_embedding_padding'] == 144
    assert seen['embedding_modules'] == {'embed':'input_embeddings'}
    assert seen['embedding_padding_modules'] == ['embed']
    assert seen['weights_mapper'] == 'legacy-mapper'
    if from_tensors:
        assert seen['embeddings'] is None


def test_loader_internal_type_error_is_preserved_without_retry(monkeypatch):
    module, Manager, _ = _load_utils(monkeypatch, '0.1.dev20073')
    calls = []
    failure = TypeError('malformed adapter tensor')

    class Model:
        @classmethod
        def from_lora_tensors(cls, lora_model_id, tensors, peft_helper, device, dtype,
                              weights_mapper, model_vocab_size=None):
            calls.append(lora_model_id)
            raise failure

    request = module.TensorLoRARequest(lora_int_id=7, peft_config={}, lora_tensors={})
    with pytest.raises(TypeError) as error:
        Manager._load_adapter(_manager(Model), request)
    assert error.value is failure
    assert calls == [7]


@pytest.mark.parametrize('target', [r'.*\.(q_proj|input_mix_weight_down)$', ['q_proj','v_proj']])
def test_strategy_preserves_target_semantics_and_caller_config(target):
    # Isolate this dependency-free real method; importing the strategy otherwise
    # requires the optional GPU engine and Ray executor before running any code.
    tree = ast.parse((ROOT/'roll/distributed/strategy/vllm_strategy.py').read_text())
    strategy = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name=='VllmStrategy')
    method = next(node for node in strategy.body if isinstance(node, ast.AsyncFunctionDef) and node.name=='add_lora')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), 'actual_vllm_strategy_add_lora', 'exec'), namespace)
    received = []

    class Receiver:
        async def add_lora(self, config):
            received.append(config)
            return [True, True]

    owner = SimpleNamespace(worker_config=SimpleNamespace(num_gpus_per_worker=2,
                            model_args=SimpleNamespace(lora_target=target)),
                            model=Receiver())
    original = {'r':8, 'target_modules':['original']}
    asyncio.run(namespace['add_lora'](owner, original))
    assert received[0]['target_modules'] == target
    assert original == {'r':8, 'target_modules':['original']}
