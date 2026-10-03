"""Forward parity with the TP2/EP8/ETP1 folding layout used for real weights."""
import os

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(os.environ.get("RUN_QWEN4_EP_TESTS") != "1", reason="requires eight CUDA ranks")
def test_expert_parallel_folding_matches_unsharded_forward():
    from megatron.core import parallel_state, tensor_parallel
    from mcore_adapter.models.converter.model_converter import ModelConverter
    from test_qwen4_exp_distributed import to_hf
    from test_qwen4_exp_model import tiny_config, make_model

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(741)
    torch.manual_seed(741)
    config = tiny_config()
    config.num_moe_experts = 16
    config.moe_router_topk = 10
    config.hf_model_type = "qwen4_exp"
    config.swiglu = True
    baseline = make_model(config).eval()
    table = baseline.decoder.layers[1].ple.ple_embedding.store
    for param in baseline.parameters():
        dist.broadcast(param.data, src=0)
    hf_weights = to_hf(config, baseline.state_dict())
    ids = torch.arange(32, device="cuda").remainder(15).add(1).unsqueeze(0)
    positions = torch.arange(32, device="cuda").unsqueeze(0)
    with torch.no_grad():
        expected = baseline(ids, positions, None)
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=8,
                                             expert_tensor_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(741)
    config.tensor_model_parallel_size = 2
    config.expert_model_parallel_size = 8
    config.expert_tensor_parallel_size = 1
    config.sequence_parallel = True
    config.moe_parallel_folding = True
    model = make_model(config).eval()
    model.decoder.layers[1].ple.ple_embedding.store = table
    converted = ModelConverter(config).get_mca_state_dict(iter(hf_weights.items()), vp_stage=0)
    result = model.load_state_dict(converted, strict=False)
    assert not result.unexpected_keys
    assert not [name for name in result.missing_keys if not name.endswith("._extra_state")]
    with torch.no_grad():
        actual = model(ids, positions, None)
        actual = tensor_parallel.gather_from_tensor_model_parallel_region(actual)
    relative = (actual.float()-expected.float()).norm()/expected.float().norm()
    print("EP_FOLDING_RELATIVE_L2", dist.get_rank(), float(relative), flush=True)
    assert relative < 0.025
    reference_replica = actual.clone()
    dist.broadcast(reference_replica, src=0)
    replica_error = (actual.float() - reference_replica.float()).norm() / reference_replica.float().norm()
    print("EP_REPLICA_RELATIVE_L2", dist.get_rank(), float(replica_error), flush=True)
    assert replica_error < 1e-5, "identical inputs/weights must agree across DP replicas"
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
