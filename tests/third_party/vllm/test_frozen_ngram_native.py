"""Run separately on an idle CUDA device with the installed native vLLM worker."""
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.mark.skipif(os.environ.get("RUN_FROZEN_NGRAM_NATIVE") != "1",
                    reason="requires an isolated CUDA process and native vLLM CuMem")
def test_native_cumem_two_cycles_preserve_tables_and_release_backbone():
    from vllm.device_allocator import get_mem_allocator_instance
    from vllm.v1.worker.gpu_worker import Worker as NativeWorker
    from roll.third_party.vllm.worker import WorkerBase
    from vllm.models.qwen3_8_flash_next.nvidia.model_state import Qwen3_8FlashNextModelState

    class Worker(WorkerBase):
        sleep = NativeWorker.sleep
        wake_up = NativeWorker.wake_up
        _get_sleep_mode_backend = NativeWorker._get_sleep_mode_backend
        get_draft_model = NativeWorker.get_draft_model

    torch.cuda.set_device(0)
    allocator = get_mem_allocator_instance()
    assert allocator.get_current_usage() == 0
    with allocator.use_memory_pool(tag="weights"):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        model.layers[1].ple = nn.Module()
        model.layers[1].ple.ple_embedding = nn.Module()
        model.layers[1].ple.ple_embedding.ngram_embedding = nn.Embedding(
            8192, 2048, dtype=torch.bfloat16, device="cuda")
        model.backbone = nn.Parameter(torch.full((8192, 2048), 3., device="cuda", dtype=torch.bfloat16))
        model.register_buffer("hash_constants", torch.arange(16, device="cuda"))
        # Native GPUModelRunner.load_model creates this state inside the
        # weights pool too, but it is not part of model.named_buffers().
        state = Qwen3_8FlashNextModelState.__new__(Qwen3_8FlashNextModelState)
        state.uses_ngram_embedding = True
        state.ngram_context_len = 2
        state.ngram_eos_token_id = 99
        state.ngram_context = torch.full((2, 2), 99, dtype=torch.int32, device="cuda")
        state.ngram_context_offsets = torch.arange(-2, 0, dtype=torch.int64, device="cuda")
    with allocator.use_memory_pool(tag="kv_cache"):
        kv_cache = torch.zeros(4 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")
    table = model.layers[1].ple.ple_embedding.ngram_embedding.weight
    expected = table.detach().cpu()
    buffers = model.hash_constants.cpu()
    worker = Worker()
    wake_events = []
    worker.model_runner = SimpleNamespace(model=model, model_state=state, get_draft_model=lambda: None,
        post_kv_cache_wake_up=lambda: wake_events.append("kv_cache"))
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen4_exp",
        text_config=SimpleNamespace(ple_layer_ids=[2])), sleep_mode_backend="cumem")
    worker.model_config = config
    worker.vllm_config = SimpleNamespace(model_config=config, lora_config=None)
    worker._sleep_mode_backend = None
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_draft_buffers = {}
    worker.custom_init_worker()
    batch = SimpleNamespace(num_reqs=2, num_reqs_after_padding=2,
        idx_mapping=torch.tensor([0, 1], device="cuda"))
    requests = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([0, 3], device="cuda")),
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[1, 2, 3, 4], [10, 11, 12, 13]], device="cuda")))
    expected_context = state._prepare_ngram_context(batch, requests).cpu()
    assert expected_context.tolist() == [[99, 99], [11, 12]]

    for cycle in range(2):
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        worker.offload_states(2)
        free_asleep = torch.cuda.mem_get_info()[0]
        snapshot = worker._frozen_ngram_sleep_state
        assert snapshot.cpu_bytes == 32 * 1024**2
        assert not worker.weight_loaded and not worker.kv_cache_loaded
        assert all(data.is_asleep for data in allocator.pointer_to_data.values())
        assert all(data.cpu_backup_tensor is None for data in allocator.pointer_to_data.values())
        assert free_asleep - free_before >= 64 * 1024**2
        worker.reload_model()
        # Exercise the native input construction, including fresh prefill and
        # a continuation. Merely restoring model parameters misses this bug.
        torch.testing.assert_close(state._prepare_ngram_context(batch, requests).cpu(),
                                   expected_context, atol=0, rtol=0)
        torch.testing.assert_close(table.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(model.hash_constants.cpu(), buffers, atol=0, rtol=0)
        assert worker.weight_loaded and not worker.kv_cache_loaded
        assert worker._frozen_ngram_sleep_state is None
        assert all(data.is_asleep == (data.tag == "kv_cache")
                   for data in allocator.pointer_to_data.values())
        # Full synchronization supplies new backbone weights; the table must
        # survive independently and native KV wake must still execute.
        with torch.no_grad():
            model.backbone.fill_(cycle + 5)
        worker.load_states()
        kv_cache.zero_()
        assert worker.kv_cache_loaded
        assert all(not data.is_asleep for data in allocator.pointer_to_data.values())
        assert torch.all(model.backbone == cycle + 5)
        assert wake_events == ["kv_cache"] * (cycle + 1)
        print({"cycle": cycle, "freed_bytes": free_asleep - free_before,
               "table_cpu_bytes": 32 * 1024**2, "native_cpu_backup_bytes": 0}, flush=True)
