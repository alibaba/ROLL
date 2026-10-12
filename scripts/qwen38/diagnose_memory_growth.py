"""Diagnostic-only wrapper around full_training_probe; no model changes.
DIAG_COLLECT=1 tests cyclic GC; DIAG_DETACH_PROBS=1 detaches the finished
MoE dispatcher probability cache. Rank 0 records active allocator provenance.
"""
import collections
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import torch
from megatron.core.tensor_parallel.random import CheckpointFunction

rank = int(os.environ['RANK'])
out = Path(sys.argv[sys.argv.index('--output') + 1])
out.mkdir(parents=True, exist_ok=True)
original = CheckpointFunction.backward
count = 0

def capture(label):
    if rank != 0:
        return
    groups = collections.defaultdict(lambda: [0, 0])
    for segment in torch.cuda.memory._snapshot()['segments']:
        for block in segment['blocks']:
            if block['state'] != 'active_allocated':
                continue
            frames = block.get('frames', [])
            key = '|'.join(f"{f.get('filename')}:{f.get('line')}:{f.get('name')}" for f in frames if f.get('filename', '').endswith('.py'))
            groups[key][0] += block['size']
            groups[key][1] += 1
    tensors = collections.Counter()
    refs = []
    # Keep only scalar metadata. Do not retain enumerated GPU tensors.
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.numel() >= 1000000:
                tensors[str((tuple(obj.shape), str(obj.dtype), obj.is_leaf, type(obj.grad_fn).__name__))] += 1
                if tuple(obj.shape) == (4096, 1, 10240) and len(refs) < 4:
                    descriptions = []
                    for ref in gc.get_referrers(obj):
                        desc = type(ref).__name__
                        if isinstance(ref, dict):
                            desc += ':' + ','.join(str(k) for k,v in ref.items() if v is obj)
                        elif hasattr(ref, 'f_code'):
                            desc += ':' + ref.f_code.co_name + ':' + str(ref.f_lineno)
                        descriptions.append(desc)
                    refs.append(descriptions)
        except Exception:
            pass
    entry = dict(label=label, completed=count, allocated=torch.cuda.memory_allocated(),
                 groups=sorted([(v[0], v[1], k) for k,v in groups.items()], reverse=True), tensors=dict(tensors), refs=refs)
    with (out/'diagnostic-rank0.jsonl').open('a') as f:
        f.write(json.dumps(entry)+'\n')
    print('DIAGNOSTIC', label, count, entry['allocated'], flush=True)

def backward(ctx, *args):
    global count
    result = original(ctx, *args)
    count += 1
    if count % 10 == 0:
        capture('completed-before-gc')
        if os.environ.get('DIAG_COLLECT') == '1':
            gc.collect()
            capture('completed-after-gc')
    return result

# Single-variable intervention: sever only the completed router graph cache.
if os.environ.get('DIAG_DETACH_PROBS') == '1':
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    original_combine = MoEAlltoAllTokenDispatcher.combine_postprocess
    def combine(self, *args, **kwargs):
        result = original_combine(self, *args, **kwargs)
        if self.probs is not None:
            self.probs = self.probs.detach()
        return result
    MoEAlltoAllTokenDispatcher.combine_postprocess = combine

CheckpointFunction.backward = staticmethod(backward)
orig_backward = torch.Tensor.backward
started = False

def tensor_backward(self, *args, **kwargs):
    global started
    if not started:
        started = True
        if rank == 0:
            torch.cuda.memory._record_memory_history(max_entries=150000)
        capture('before-backward')
    try:
        return orig_backward(self, *args, **kwargs)
    except Exception:
        capture('exception')
        raise

torch.Tensor.backward = tensor_backward
runpy.run_path(str(Path(__file__).with_name('full_training_probe.py')), run_name='__main__')
