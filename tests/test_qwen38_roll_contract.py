from pathlib import Path


def test_megatron_strategy_forwards_cpu_optimizer_configuration():
    source = Path(__file__).parents[1].joinpath("roll/distributed/strategy/megatron_strategy.py").read_text()
    block = source[source.index("optimizer_config = OptimizerConfig("):source.index("self.optimizer:", source.index("optimizer_config = OptimizerConfig("))]
    assert "optimizer_cpu_offload=self.megatron_train_args.optimizer_cpu_offload" in block
    assert "optimizer_offload_fraction=self.megatron_train_args.optimizer_offload_fraction" in block


def test_flash_next_adapter_is_registered():
    source = Path(__file__).parents[1].joinpath("mcore_adapter/src/mcore_adapter/models/__init__.py").read_text()
    assert "qwen4_exp" in source
