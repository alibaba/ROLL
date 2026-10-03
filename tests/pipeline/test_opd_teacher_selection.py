"""An OPD student's LoRA base cannot replace its explicitly configured teacher."""
import importlib
from types import SimpleNamespace

import pytest


class BeforeExternalResources(Exception):
    pass


@pytest.mark.parametrize('kind', ['rlvr', 'agentic'])
@pytest.mark.parametrize('mode,lora,enabled,expected', [
    ('pure_opd', True, True, True),
    ('mixed_opd', True, True, True),
    ('pure_opd', False, True, True),
    ('mixed_opd', False, True, True),
    ('rl', True, True, False),
    ('rl', False, True, True),
    ('rl', False, False, False),
])
def test_pipeline_selects_configured_teacher_for_opd(monkeypatch, kind, mode, lora, enabled, expected):
    module = importlib.import_module(f'roll.pipeline.{kind}.{kind}_pipeline')
    pipeline_type = module.RLVRPipeline if kind == 'rlvr' else module.AgenticPipeline
    config = SimpleNamespace(
        enable_reference=enabled, is_pure_opd=mode == 'pure_opd', use_opd=mode == 'mixed_opd',
        actor_train=SimpleNamespace(model_args=SimpleNamespace(lora_target=['linear_qkv'] if lora else None)),
        max_steps=2, set_max_steps=lambda **kwargs: None,
        init_kl_coef=0.0, target_kl=0.1, kl_horizon=100,
    )
    # Base initialization starts Ray resources, trackers and checkpoint IO.
    # Preserve its config assignment; stop before tokenizer/model/dataset IO.
    def initialize_base(self, pipeline_config):
        self.pipeline_config = pipeline_config

    def stop(*args, **kwargs):
        raise BeforeExternalResources()

    monkeypatch.setattr(module.BasePipeline, '__init__', initialize_base)
    monkeypatch.setattr(module, 'default_tokenizer_provider' if kind == 'rlvr' else 'get_kl_controller', stop)
    pipeline = object.__new__(pipeline_type)
    with pytest.raises(BeforeExternalResources):
        pipeline_type.__init__(pipeline, config)
    assert pipeline.use_ref_model is expected
