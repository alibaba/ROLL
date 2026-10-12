"""Collective frozen-asset coverage, plus opt-in real PP2/VPP2 model stages."""
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_qwen4_exp_asset_lifecycle import AssetStage, checkpoint_fixture, load_qwen4_module


def _exercise_resume(models, source, saved, lifecycle, load, save):
    loaded = load(source)
    save(saved)
    record = json.loads((saved / lifecycle.EXTERNAL_ASSET_METADATA_NAME).read_text())
    assert set(record["manifests"]) == {"1", "7"}
    relocated = source.with_name("relocated")
    if dist.get_rank() == 0:
        shutil.move(source, relocated)
    dist.barrier()
    assert load(saved, external_asset_path=relocated) == loaded

    original = json.dumps(record)
    for mutation in ("missing", "forged"):
        if dist.get_rank() == 0:
            corrupted = json.loads(original)
            if mutation == "missing":
                del corrupted["manifests"]["7"]
            else:
                corrupted["manifests"]["7"]["index_sha256"] = "0" * 64
            (saved / lifecycle.EXTERNAL_ASSET_METADATA_NAME).write_text(json.dumps(corrupted))
        dist.barrier()
        with pytest.raises(ValueError, match="coverage|manifest mismatch"):
            load(saved, external_asset_path=relocated)
        # If only rank 1 rejects its forged manifest, rank 0 would hang here or
        # continue into model/optimizer loading. Both ranks must reject it.
        dist.barrier()
    if dist.get_rank() == 0:
        (saved / lifecycle.EXTERNAL_ASSET_METADATA_NAME).write_text(original)
    dist.barrier()
    assert load(saved, external_asset_path=relocated) == loaded


def _cpu_worker(rank, directory):
    directory = Path(directory)
    dist.init_process_group(
        "gloo", init_method=(directory / "rendezvous").as_uri(), rank=rank, world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        lifecycle = load_qwen4_module("asset_lifecycle")
        models = [AssetStage([1] if rank == 0 else [], global_layers=(1, 7)),
                  AssetStage([7] if rank == 1 else [], global_layers=(1, 7))]
        _exercise_resume(
            models, directory / "source", directory / "saved", lifecycle,
            lambda path, **kwargs: lifecycle.restore_ngram_asset_models(models, path, **kwargs),
            lambda path: lifecycle.persist_ngram_asset_models(models, path),
        )
        incomplete = [AssetStage([1] if rank == 0 else [], global_layers=(1, 7))]
        with pytest.raises(ValueError, match="coverage"):
            lifecycle.restore_ngram_asset_models(incomplete, directory / "relocated")
        if rank == 1:
            delattr(models[1], "_qwen4_ngram_asset_record")
        with pytest.raises((ValueError, RuntimeError), match="before frozen n-gram assets are attached"):
            lifecycle.persist_ngram_asset_models(models, directory / "incomplete")
        assert not (directory / "incomplete" / lifecycle.EXTERNAL_ASSET_METADATA_NAME).exists()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the CPU Gloo backend")
def test_collective_pp_and_virtual_stage_coverage(tmp_path):
    checkpoint_fixture(tmp_path / "source", layer_indices=(1, 7))
    mp.spawn(_cpu_worker, args=(str(tmp_path),), nprocs=2, join=True)


@pytest.fixture(scope="module")
def actual_pp_environment():
    if os.environ.get("RUN_QWEN4_ASSET_PP_TESTS") != "1":
        pytest.skip("requires two real CUDA ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Keep the torchrun rendezvous alive across parameterized cases. Reusing
    # its store after destroying/recreating the world group races between ranks.
    dist.init_process_group("nccl", timeout=timedelta(seconds=90),
                            device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    yield
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_QWEN4_ASSET_PP_TESTS") != "1", reason="requires two real CUDA ranks")
@pytest.mark.parametrize("virtual_size", [None, 2])
@pytest.mark.parametrize("num_layers", [8, 12])
def test_actual_pp2_model_stages_keep_complete_asset_identity(actual_pp_environment, virtual_size, num_layers):
    from megatron.core import parallel_state, tensor_parallel
    from mcore_adapter.models.model_factory import VirtualModels
    from mcore_adapter.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
    from mcore_adapter.models.qwen4_exp import asset_lifecycle as lifecycle
    from test_qwen4_exp_model import tiny_config

    directory = Path(os.environ["QWEN4_ASSET_PP_DIRECTORY"]) / f"layers-{num_layers}-vpp-{virtual_size or 1}"
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=virtual_size,
        )
        tensor_parallel.model_parallel_cuda_manual_seed(1813)
        config = tiny_config()
        config.num_layers = num_layers
        config.layer_types = config._derive_layer_types()
        config.pipeline_model_parallel_size = 2
        config.virtual_pipeline_model_parallel_size = virtual_size
        config.ple_layer_ids = [2, 8]
        config.ple_embed_dim = 8
        config.ngram_vocab_size_base = 3
        if dist.get_rank() == 0:
            directory.mkdir(parents=True)
            checkpoint_fixture(directory / "source", layer_indices=(1, 7))
        dist.barrier()
        models = VirtualModels(Qwen4ExpModel, config)
        # Twelve layers split into six-layer stages or three-layer chunks:
        # neither boundary aligns with the four-layer attention period.
        expected_layers = {
            (8, None): ([[0, 1, 2, 3]], [[4, 5, 6, 7]]),
            (8, 2): ([[0, 1], [4, 5]], [[2, 3], [6, 7]]),
            (12, None): ([[0, 1, 2, 3, 4, 5]], [[6, 7, 8, 9, 10, 11]]),
            (12, 2): ([[0, 1, 2], [6, 7, 8]], [[3, 4, 5], [9, 10, 11]]),
        }[(num_layers, virtual_size)][dist.get_rank()]
        assert [[layer.layer_number - 1 for layer in model.decoder.layers] for model in models] == expected_layers
        for model, indices in zip(models, expected_layers):
            for layer, index in zip(model.decoder.layers, indices):
                expected_class = "Qwen4ExpQSAAttention" if index in (3, 7, 11) else "Qwen4ExpGatedDeltaNet"
                assert type(layer.self_attention).__name__ == expected_class, index
        local_ple_layers = [[layer.layer_number - 1 for layer in model.decoder.layers if hasattr(layer, "ple")]
                            for model in models]
        if virtual_size == 2 and num_layers == 12:
            assert local_ple_layers == ([[1], [7]] if dist.get_rank() == 0 else [[], []])
        elif virtual_size == 2:
            assert local_ple_layers == ([[1], []] if dist.get_rank() == 0 else [[], [7]])
        else:
            assert local_ple_layers == ([[1]] if dist.get_rank() == 0 else [[7]])
        _exercise_resume(models, directory / "source", directory / "saved", lifecycle,
                         models.load_external_assets, models.save_external_assets)
        # Run the real MCA serialization hook as well as the direct asset hook.
        models.save_pretrained(directory / "mca-saved")
        assert set(json.loads((directory / "mca-saved" / lifecycle.EXTERNAL_ASSET_METADATA_NAME).read_text())["manifests"]) == {"1", "7"}
        print("PP_ASSET_COVERAGE_OK", dist.get_rank(), virtual_size, local_ple_layers, flush=True)
    finally:
        dist.barrier()
        parallel_state.destroy_model_parallel()
