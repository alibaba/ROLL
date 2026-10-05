"""DDP initialization must release original weights before allocating gradients."""
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_DDP_MEMORY_TESTS") != "1", reason="requires Megatron CUDA")
def test_ddp_remapping_has_only_two_parameter_sized_allocations(tmp_path):
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer import MegatronModule, TransformerConfig

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4,
                                   bf16=True, params_dtype=torch.bfloat16)
        model = MegatronModule(config)
        for index in range(4):
            model.register_parameter(f"weight_{index}", torch.nn.Parameter(
                torch.full((16 * 1024 * 1024,), index + 1, device="cuda", dtype=torch.bfloat16)))
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        wrapped = DistributedDataParallel(config, DistributedDataParallelConfig(
            grad_reduce_in_fp32=False, use_distributed_optimizer=True, overlap_grad_reduce=False,
        ), model)
        extra_peak = torch.cuda.max_memory_allocated() - baseline
        print("DDP_INIT_PEAK", dict(parameter_bytes=parameter_bytes, extra_peak=extra_peak), flush=True)
        assert extra_peak < parameter_bytes * 1.25, "initialization retained original and remapped weights with gradients"
        for index, param in enumerate(model.parameters()):
            assert bool((param == index + 1).all()), "remapping changed parameter values"
            assert param.main_grad.dtype == torch.bfloat16
        wrapped.zero_grad_buffer()
        sum(p[:8].float().sum() for p in model.parameters()).backward()
        wrapped.finish_grad_sync()
        for param in model.parameters():
            torch.testing.assert_close(param.main_grad[:8], torch.ones_like(param.main_grad[:8]))
            assert not bool(param.main_grad[8:].any())
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
