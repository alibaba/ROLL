from dataclasses import dataclass
import yaml
from pathlib import Path

from roll.third_party.megatron.optimizer_config import build_optimizer_config


def test_megatron_strategy_forwards_cpu_optimizer_configuration():
    @dataclass
    class Config:
        lr: float
        optimizer_cpu_offload: bool = False
        optimizer_offload_fraction: float = 0.0

    class Args:
        optimizer_cpu_offload = True
        optimizer_offload_fraction = 1.0

    config = build_optimizer_config(Config, {"lr": 0.01}, Args())
    assert config.optimizer_cpu_offload is True
    assert config.optimizer_offload_fraction == 1.0


def test_flash_next_adapter_is_registered():
    source = Path(__file__).parents[1].joinpath("mcore_adapter/src/mcore_adapter/models/__init__.py").read_text()
    assert "qwen4_exp" in source


def test_flash_next_opd_configs_do_not_drop_short_teacher_rollouts():
    root = Path(__file__).parents[1]
    for name in ("opd_lora.yaml", "opd_backbone.yaml"):
        config = yaml.safe_load((root / "scripts/qwen38/configs" / name).read_text())
        assert config["max_len_mask"] is False
