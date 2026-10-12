"""Bounded actual HF -> Megatron conversion and layer parity probe."""
import ast
import math
import types
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from torch import nn
from torch.nn import functional as F


def load_pinned_hf():
    path = Path(os.environ.get("QWEN4_PINNED_SOURCE", "/data_nvme/workspace/roll-qwen38-validation/modeling_qwen4_exp.py"))
    source = path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == "2a44aeadb215acbb5c75939fcc97e9f14bccff5a51c232826427594993f6a760"
    tree = ast.parse(source)
    keep = {"Qwen4ExpTextRMSNorm", "Qwen4ExpTextRMSNormGated", "Qwen4ExpTextGatedDeltaNet",
            "Qwen4ExpTextQSAIndexer", "Qwen4ExpTextAttention", "Qwen4ExpTextMLP", "Qwen4ExpTextExperts",
            "Qwen4ExpTextTopKRouter", "Qwen4ExpTextSparseMoeBlock", "Qwen4ExpTextGatedResidual",
            "Qwen4ExpTextNGramEmbedding", "Qwen4ExpTextPLELayer", "Qwen4ExpTextDecoderLayer"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) and node.name in keep:
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                    child.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t,ast.Name) and t.id.startswith('_') and t.id.isupper() for t in node.targets):
            nodes.append(node)
    namespace = dict(torch=torch, nn=nn, F=F, math=math, ACT2FN={"silu":F.silu, "sigmoid":torch.sigmoid}, GradientCheckpointingLayer=nn.Module, is_torchdynamo_exporting=lambda: False, is_torchdynamo_compiling=lambda: False, use_kernel_func_from_hub_with_fallback=lambda *a,**k: (lambda f:f), use_kernelized_func=lambda *a,**k: (lambda f:f), use_kernel_forward_from_hub=lambda *a,**k: (lambda f:f), force_accelerate_hooks=lambda *a,**k: (lambda f:f))
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    namespace['ALL_ATTENTION_FUNCTIONS'] = ALL_ATTENTION_FUNCTIONS
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    namespace['__file__'] = str(path)
    return types.SimpleNamespace(**namespace)

hfmod = load_pinned_hf()
Qwen4ExpTextConfig = types.SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mcore_adapter/tests'))
from test_qwen4_exp_model import tiny_config, make_model
from mcore_adapter.models.converter.model_converter import ModelConverter


def main():
    print('HF_SHA256', hashlib.sha256(Path(hfmod.__file__).read_bytes()).hexdigest(), flush=True)
    print('HF_PATH', hfmod.__file__, flush=True)
    config = tiny_config()
    config.hf_model_type = "qwen4_exp"
    config.swiglu = True
    config.gdn_output_gate_type = "sigmoid"
    torch.manual_seed(725)
    h = Qwen4ExpTextConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
        hidden_act='silu', output_gate_type='sigmoid', rms_norm_eps=1e-6, attention_dropout=0.,
        _attn_implementation='eager',
        attention_bias=False, norm_topk_prob=True, seed=0,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        layer_types=['linear_attention']*3+['full_attention'], linear_conv_kernel_dim=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=6, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64, hc_count=4, hc_lowrank=16,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16,
        indexer_compress_ratio=4, ple_layer_ids=[2], ple_embed_dim=128, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=16,
        make_ngram_vocab_size_divisible_by=1, eos_token_id=0,
        rope_parameters={'rope_type':'default','rope_theta':10000.0,'partial_rotary_factor':0.25})
    print('HF_CONFIG', h, flush=True)
    assert config.layernorm_epsilon == h.rms_norm_eps, (
        f"parity requires identical RMS epsilon: MCA={config.layernorm_epsilon}, HF={h.rms_norm_eps}"
    )
    reference = nn.Module()
    reference.embed_tokens = nn.Embedding(h.vocab_size,h.hidden_size)
    reference.layers = nn.ModuleList([hfmod.Qwen4ExpTextDecoderLayer(h,i) for i in range(4)])
    reference.hyper_connection_mixer = hfmod.Qwen4ExpTextGatedResidual(h,use_combine=False)
    with torch.no_grad():
        for name,param in reference.named_parameters():
            if name.endswith('A_log'): param.uniform_(-1,1)
            elif name.endswith('linear_attn.norm.weight'): param.uniform_(0.5,1.5)
            elif 'norm' in name: param.uniform_(-0.2,0.2)
            else: param.normal_(std=0.03)
    reference = reference.bfloat16().cuda()
    source = {'model.language_model.'+k: v.detach().cpu().clone() for k,v in reference.state_dict().items()}
    source = {k.replace('.ngram_embedding.weight','.ngram_embedding.shard_0'):v for k,v in source.items()}
    source['lm_head.weight'] = torch.randn(256,128).bfloat16()
    print('SOURCE_SHAPES', {k:list(v.shape) for k,v in source.items()}, flush=True)
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29753', RANK='0', WORLD_SIZE='1')
    dist.init_process_group('nccl',rank=0,world_size=1)
    parallel_state.initialize_model_parallel(1,1)
    tensor_parallel.model_parallel_cuda_manual_seed(717)
    model = make_model(config)
    converter = ModelConverter(config)
    converted = converter.get_mca_state_dict(iter(source.items()), vp_stage=0)
    expected = {k:v for k,v in model.state_dict().items() if isinstance(v,torch.Tensor) and not k.endswith('_extra_state')}
    print('MISSING', sorted(set(expected)-set(converted)), flush=True)
    print('UNEXPECTED', sorted(set(converted)-set(expected)), flush=True)
    mismatches={k:[list(converted[k].shape),list(expected[k].shape)] for k in set(expected)&set(converted) if converted[k].shape!=expected[k].shape}
    print('SHAPE_MISMATCH',mismatches,flush=True)
    assert not set(expected)-set(converted)
    assert not set(converted)-set(expected)
    assert not mismatches
    result=model.load_state_dict(converted,strict=False)
    assert not result.unexpected_keys
    assert not [k for k in result.missing_keys if not k.endswith('_extra_state')]
    for k,v in converted.items(): torch.testing.assert_close(model.state_dict()[k].cpu(),v,atol=0,rtol=0)
    print('LOAD_PASS',len(converted),flush=True)
    print('SHARED_WIDTH',config.moe_shared_expert_intermediate_size,tuple(model.decoder.layers[0].mlp.shared_experts.linear_fc1.weight.shape),flush=True)
    model.eval()
    reference.eval()
    from mcore_adapter.models.qwen4_exp.ngram_embedding import TensorNGramStore
    model.decoder.layers[1].ple.ple_embedding.store = TensorNGramStore(
        reference.layers[1].ple.ple_embedding.ngram_embedding.weight.detach().cpu()
    )
    x = torch.randn(2,32,128,device='cuda',dtype=torch.bfloat16)*0.5
    metrics = {}
    def compare(name, actual, expected):
        a,b = actual.float(),expected.float()
        metrics[name] = {'max_abs':float((a-b).abs().max()), 'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-9)), 'reference_norm':float(b.norm())}
        print('PARITY',name,metrics[name],flush=True)
    with torch.no_grad():
        for i in [0,2]:
            got = model.decoder.layers[i].self_attention(x.transpose(0,1).contiguous(),attention_mask=None)[0].transpose(0,1)
            want = reference.layers[i].linear_attn(x)
            compare(f'gdn{i}',got,want)
        for i in [0,3]:
            got = model.decoder.layers[i].mlp(x.transpose(0,1).contiguous())[0].transpose(0,1)
            want = reference.layers[i].mlp(x)
            compare(f'moe{i}',got,want)
        stream=torch.randn(2,32,512,device='cuda',dtype=torch.bfloat16)*0.5
        got=model.decoder.layers[0](stream.transpose(0,1).contiguous(),attention_mask=None)[0].transpose(0,1)
        want=reference.layers[0](stream,position_embeddings=None)
        compare('gdn_gr_layer0',got,want)
        # Independent pinned decoder stack, including PLE and sparse QSA.
        # Compare logits as well as residual increments so a large skip path
        # cannot hide a broken attention/MLP branch.
        ids = torch.arange(32, device='cuda').remainder(15).add(1).unsqueeze(0)
        angles = model.rotary_pos_emb(32)[:, 0, 0].unsqueeze(0)
        positions = (angles.cos().bfloat16(), angles.sin().bfloat16())
        causal = torch.zeros(1, 1, 32, 32, device='cuda', dtype=torch.bfloat16)
        causal.masked_fill_(torch.ones(32, 32, device='cuda', dtype=torch.bool).triu(1),
                            torch.finfo(torch.bfloat16).min)
        stream_hf = reference.embed_tokens(ids).repeat(1, 1, h.hc_count)
        stream_mca = stream_hf.transpose(0, 1).contiguous()
        for i, layer in enumerate(reference.layers):
            hf_input, mca_input = stream_hf, stream_mca
            stream_hf = layer(hf_input, positions, attention_mask=causal, ple_input_ids=ids)
            stream_mca = model.decoder._run_layers(
                i, i+1, mca_input, ids, torch.ones_like(ids, dtype=torch.bool),
                None, None, None, angles.squeeze(0).unsqueeze(1).unsqueeze(1))
            compare(f'decoder{i}', stream_mca.transpose(0, 1), stream_hf)
        mixed_hf = reference.hyper_connection_mixer(stream_hf)
        assert mixed_hf.shape == (1, 32, h.hidden_size)
        wanted_logits = F.linear(mixed_hf, source['lm_head.weight'].cuda())
        actual_logits = model(ids, torch.arange(32, device='cuda').unsqueeze(0), None)
        compare('full_logits', actual_logits, wanted_logits)
    print('METRICS',json.dumps(metrics),flush=True)
    for name, metric in metrics.items():
        assert metric['relative_l2'] < 0.03, (name, metric)
    index=json.loads(Path('/data_hdd/Qwen3.8-Flash-Next/model.safetensors.index.json').read_text())['weight_map']
    print('REAL_STALE_NORMS',[k for k in index if '.input_layernorm.' in k or '.post_attention_layernorm.' in k],flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

if __name__=='__main__': main()
