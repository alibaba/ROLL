"""CPU regressions for online Qwen4 expert export; requires installed Megatron.

RUN_QWEN4_STREAMING_TESTS=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q <this file>
Add RUN_QWEN4_NATIVE_VLLM_TESTS=1 to exercise the installed vLLM loader.
"""

import copy
import os
import re
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN4_STREAMING_TESTS") != "1", reason="requires the installed Megatron converter"
)


def converter(ep_rank=0, num_experts=512, ep_size=8):
    from mcore_adapter.models.converter.model_converter import ModelConverter

    config = SimpleNamespace(
        hf_model_type="qwen4_exp", num_moe_experts=num_experts, expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None, pipeline_model_parallel_layout=None,
        num_layers=1, hidden_size=8, moe_ffn_hidden_size=16, swiglu=True,
        moe_grouped_gemm=True, transformer_impl="transformer_engine",
        account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False,
        tie_embeddings_and_output_weights=False,
    )
    result = ModelConverter(config, tensor_model_parallel_rank=0, pipeline_model_parallel_rank=0,
                            expert_model_parallel_rank=ep_rank, expert_tensor_parallel_rank=0,
                            to_hf=True, efficient_mode=True)
    # Isolate pre-existing checkpoint accumulators, while executing the actual converter.
    result.template = copy.deepcopy(result.template)
    result.template._expert_buffer.clear()
    result.template.release()
    return result


def expert_weights(expert_id, version=0):
    # The small integer pattern remains exact in FP32 and differs in every row/column.
    gate_up = torch.arange(256, dtype=torch.float32).reshape(32, 8) + expert_id * 512 + version * 1_000_000
    down = torch.arange(128, dtype=torch.float32).reshape(8, 16) + expert_id * 512 + version * 1_000_000
    return gate_up, down


def exported_weights(item, local_e, global_e, version=0, dtype=torch.float32, **kwargs):
    for fc, tensor in zip((1, 2), expert_weights(global_e, version)):
        name = f"decoder.layers.0.mlp.experts.linear_fc{fc}.weight{local_e}"
        yield item.convert_to_hf({name: [tensor.to(dtype)]}, vp_stage=0, **kwargs)


def test_ep8_exports_all_512_experts_immediately_without_layer_accumulation():
    seen = set()
    largest_output = 0
    for ep_rank in range(8):
        item = converter(ep_rank)
        for local_e in range(64):
            global_e = ep_rank * 64 + local_e
            for fc, output in enumerate(exported_weights(item, local_e, global_e, expert_format="per_expert")):
                projection = ("gate_up_proj", "down_proj")[fc]
                name = f"model.language_model.layers.0.mlp.experts.{global_e}.{projection}.weight"
                assert set(output) == {name}, "every input must emit one expert immediately"
                assert name not in seen
                seen.add(name)
                torch.testing.assert_close(output[name], expert_weights(global_e)[fc], rtol=0, atol=0)
                largest_output = max(largest_output, sum(w.numel() * w.element_size() for w in output.values()))
                assert not item.template._expert_buffer
                assert not item.dist_converter.weights_waiting_for_convert
    assert len(seen) == 1024
    assert largest_output == 1024


def test_per_expert_mode_does_not_change_checkpoint_stacking_or_mix_versions():
    item = converter(num_experts=2, ep_size=1)
    # Keep an incomplete checkpoint export while an independent streaming call occurs.
    assert list(exported_weights(item, 0, 0)) == [{}, {}]
    streamed = list(exported_weights(item, 1, 1, version=1, expert_format="per_expert"))
    assert all(len(output) == 1 for output in streamed)
    checkpoint = list(exported_weights(item, 1, 1))
    for fc, output in enumerate(checkpoint):
        name = f"model.language_model.layers.0.mlp.experts.{('gate_up_proj', 'down_proj')[fc]}"
        expected = torch.stack([expert_weights(0)[fc], expert_weights(1)[fc]])
        torch.testing.assert_close(output[name], expected, rtol=0, atol=0)
    assert not item.template._expert_buffer


def test_unknown_expert_format_fails_before_buffering():
    item = converter()
    with pytest.raises(ValueError, match="expert_format"):
        list(exported_weights(item, 0, 0, expert_format="typo"))
    assert not item.template._expert_buffer


def test_streaming_mode_is_call_local_even_with_shared_registered_template():
    first = converter(num_experts=2, ep_size=1)
    second = converter(num_experts=2, ep_size=1)
    second.template = first.template
    assert all(list(exported_weights(first, 0, 0, expert_format="per_expert")))
    assert list(exported_weights(second, 0, 0)) == [{}, {}]
    assert all(list(exported_weights(second, 1, 1)))


def test_update_buffer_flushes_keep_per_expert_format(monkeypatch):
    from roll.third_party.megatron import model_update

    item = converter(num_experts=4, ep_size=1)
    monkeypatch.setattr(model_update.mpu, "get_expert_tensor_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_virtual_pipeline_model_parallel_rank", lambda: 0)
    weights = [(f"decoder.layers.0.mlp.experts.linear_fc1.weight{e}", expert_weights(e)[0]) for e in range(4)]
    outputs = list(model_update._gather_hf_weights(item, weights, buffer_size=1024, expert_format="per_expert"))
    assert [len(batch) for batch in outputs] == [1, 1, 1, 1]
    assert [name for batch in outputs for name, _ in batch] == [
        f"model.language_model.layers.0.mlp.experts.{e}.gate_up_proj.weight" for e in range(4)
    ]
    assert all(sum(w.numel() * w.element_size() for _, w in batch) <= 1024 for batch in outputs)
    assert not item.template._expert_buffer


def test_online_update_selects_streaming_without_changing_checkpoint_default(monkeypatch):
    from roll.third_party.megatron import model_update

    item = converter(num_experts=4, ep_size=1)
    monkeypatch.setattr(model_update.mpu, "get_expert_tensor_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_virtual_pipeline_model_parallel_rank", lambda: 0)
    weights = [("decoder.layers.0.mlp.experts.linear_fc1.weight0", expert_weights(0)[0])]
    outputs = list(model_update._gather_hf_weights(item, weights, buffer_size=1024))
    assert len(outputs) == 1 and len(outputs[0]) == 1
    assert outputs[0][0][0] == "model.language_model.layers.0.mlp.experts.0.gate_up_proj.weight"
    assert list(exported_weights(item, 0, 0)) == [{}, {}]


def test_oversized_expert_fails_before_exceeding_gather_budget(monkeypatch):
    from roll.third_party.megatron import model_update

    item = converter(num_experts=4, ep_size=1)
    monkeypatch.setattr(model_update.mpu, "get_expert_tensor_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_virtual_pipeline_model_parallel_rank", lambda: 0)
    weights = [("decoder.layers.0.mlp.experts.linear_fc1.weight0", expert_weights(0)[0])]
    with pytest.raises(ValueError, match="expert.*buffer_size"):
        list(model_update._gather_hf_weights(item, weights, buffer_size=512, expert_format="per_expert"))
    assert not item.template._expert_buffer


def test_ep8_update_gather_preserves_every_expert_with_bounded_batches(monkeypatch):
    from roll.third_party.megatron import model_update

    item = converter()
    ep_group = object()
    monkeypatch.setattr(model_update.mpu, "get_expert_model_parallel_group", lambda: ep_group)
    monkeypatch.setattr(model_update.mpu, "get_expert_tensor_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(model_update.mpu, "get_virtual_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_update.dist, "get_world_size", lambda group: 8 if group is ep_group else 1)

    # Replace only the communication boundary with the other seven deterministic rank inputs.
    def gather_names(outputs, names, group):
        for rank in range(8):
            outputs[rank] = [re.sub(r"(?<=experts\.)\d+", lambda m: str(int(m[0]) + rank * 64), name)
                             for name in names]

    def gather_tensors(outputs, tensor, group, async_op):
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 64 * 512)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(model_update.dist, "all_gather_object", gather_names)
    monkeypatch.setattr(model_update.dist, "all_gather", gather_tensors)
    weights = [(f"decoder.layers.0.mlp.experts.linear_fc{fc + 1}.weight{e}", expert_weights(e)[fc])
               for e in range(64) for fc in range(2)]
    seen = set()
    for batch in model_update._gather_hf_weights(item, weights, buffer_size=8192, expert_format="per_expert"):
        assert batch and sum(w.numel() * w.element_size() for _, w in batch) <= 8192
        for name, weight in batch:
            assert name not in seen
            seen.add(name)
            expert_id = int(re.search(r"experts\.(\d+)", name)[1])
            fc = 0 if ".gate_up_proj." in name else 1
            torch.testing.assert_close(weight, expert_weights(expert_id)[fc], rtol=0, atol=0)
        assert not item.template._expert_buffer
    assert len(seen) == 1024


@pytest.mark.skipif(os.environ.get("RUN_QWEN4_NATIVE_VLLM_TESTS") != "1", reason="requires installed vLLM loader")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_vllm_tp8_loader_reassembles_streamed_ep8_experts(dtype):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    # Native loader methods and parameters, without constructing a GPU compute backend.
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    loaders = []
    for tp_rank in range(8):
        parallel = FusedMoEParallelConfig(tp_size=8, pcp_size=1, dp_size=1, ep_size=1,
            tp_rank=tp_rank, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1, use_ep=False,
            all2all_backend="allgather_reducescatter", enable_eplb=False)
        manager = ExpertMapManager(max_num_batched_tokens=1, top_k=1, global_num_experts=512,
            num_redundant_experts=0, num_expert_group=None, moe_parallel_config=parallel,
            placement_strategy="linear", enable_eplb=False)
        layer = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.layer_name = "model.language_model.layers.0.mlp.experts"
        layer.moe_config = SimpleNamespace(num_experts=512, num_logical_experts=512,
            tp_rank=tp_rank, moe_parallel_config=parallel, is_act_and_mul=True)
        layer.expert_map_manager = manager
        layer.quant_config = None
        layer.quant_method = None
        layer.ckpt_gate_proj_name = "gate_proj"
        layer.ckpt_up_proj_name = "up_proj"
        layer.ckpt_down_proj_name = "down_proj"
        layer.lora_base_layer_prefix = ""
        layer.is_fused_checkpoint_transposed = False
        layer.w13_weight = torch.nn.Parameter(torch.full((512, 4, 8), torch.nan, dtype=dtype), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.full((512, 8, 2), torch.nan, dtype=dtype), requires_grad=False)
        layer.w13_weight.weight_loader = layer.weight_loader
        layer.w2_weight.weight_loader = layer.weight_loader
        loaders.append(layer)

    for ep_rank in range(8):
        item = converter(ep_rank)
        for local_e in range(64):
            global_e = ep_rank * 64 + local_e
            for output in exported_weights(item, local_e, global_e, dtype=dtype, expert_format="per_expert"):
                assert len(output) == 1
                name, tensor = next(iter(output.items()))
                suffix = name.removeprefix("model.language_model.layers.0.mlp.experts.")
                for layer in loaders:
                    assert list(layer.load_weights([(suffix, tensor)])), name

    for e in range(512):
        actual_gate = torch.cat([layer.w13_weight[e, :2] for layer in loaders])
        actual_up = torch.cat([layer.w13_weight[e, 2:] for layer in loaders])
        actual_down = torch.cat([layer.w2_weight[e] for layer in loaders], dim=1)
        expected_gate_up, expected_down = expert_weights(e)
        torch.testing.assert_close(torch.cat([actual_gate, actual_up]), expected_gate_up.to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(actual_down, expected_down.to(dtype), rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
