"""Keep generation request identity across batching and actual rollout dumps."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize('with_tensor', [False, True])
def test_request_metadata_survives_postprocess_concat_reorder_and_dump(tmp_path, monkeypatch, with_tensor):
    pytest.importorskip('ray')
    from roll.distributed.scheduler.protocol import DataProto
    from roll.distributed.scheduler.user_defined_rollout_loop import postprocess_output_data
    from roll.pipeline.rlvr import utils

    def response(rid, tokens):
        meta = dict(request_id=rid, generation_config=dict(temperature=.8, num_return_sequences=2),
                    eos_token_id=[9], pad_token_id=0, output_token_ids=tokens,
                    finish_reasons=['stop', 'stop'])
        if with_tensor:
            meta['unrelated_tensor'] = torch.ones(1)
        request = DataProto.from_dict(tensors=dict(input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long), position_ids=torch.arange(3).unsqueeze(0)),
            non_tensors=dict(prompt=np.array(['same prompt'], dtype=object)), meta_info=meta)
        output = postprocess_output_data(request, DataProto(meta_info=meta), sequence_length=16)
        output.batch['scores'] = torch.zeros(2)
        return output

    first_tokens, second_tokens = [[4, 9], [5, 6, 9]], [[7, 8, 9], [8, 9]]
    first, second = response('request-a', first_tokens), response('request-b', second_tokens)
    batch = DataProto.concat([first, second])
    batch.reorder(torch.tensor([2, 0, 3, 1]))
    processes, factory = [], utils.multiprocessing.Process
    def tracked_process(*args, **kwargs):
        p = factory(*args, **kwargs)
        processes.append(p)
        return p
    monkeypatch.setattr(utils.multiprocessing, 'Process', tracked_process)
    class Tokenizer:
        def batch_decode(self, values, **kwargs):
            return ['decoded'] * len(values)
    utils.dump_rollout_to_specific_path(str(tmp_path), 3, batch, Tokenizer())
    for p in processes:
        p.join(timeout=10)
        assert p.exitcode == 0
    result = json.loads((tmp_path/'rollout_dump_data.step_3.jsonl').read_text())
    records = [json.loads(s) for s in result['sampling_params']]
    assert [r['request_id'] for r in records] == ['request-b', 'request-a', 'request-b', 'request-a']
    assert [r['output_token_ids'] for r in records] == [second_tokens[0], first_tokens[0], second_tokens[1], first_tokens[1]]
    assert all(r['generation_config'] == dict(temperature=.8, num_return_sequences=2) for r in records)
    assert [r['finish_reasons'] for r in records] == ['stop', 'stop', 'stop', 'stop']
    assert all('unrelated_tensor' not in r for r in records)
    assert result['global_step'] == [3] * 4


def test_custom_rollout_without_sample_metadata_keeps_dump_fallback(tmp_path, monkeypatch):
    pytest.importorskip('ray')
    from roll.distributed.scheduler.protocol import DataProto
    from roll.pipeline.rlvr import utils

    data = DataProto.from_dict(
        tensors=dict(responses=torch.tensor([[4, 9], [5, 9]]), scores=torch.tensor([1., -1.])),
        meta_info=dict(request_id='custom', generation_config=dict(temperature=.5)),
    )
    processes, factory = [], utils.multiprocessing.Process
    def tracked_process(*args, **kwargs):
        p = factory(*args, **kwargs)
        processes.append(p)
        return p
    monkeypatch.setattr(utils.multiprocessing, 'Process', tracked_process)
    class Tokenizer:
        def batch_decode(self, values, **kwargs):
            return ['first', 'second']
    utils.dump_rollout_to_specific_path(str(tmp_path), 4, data, Tokenizer())
    for p in processes:
        p.join(timeout=10)
        assert p.exitcode == 0
    result = json.loads((tmp_path / 'rollout_dump_data.step_4.jsonl').read_text())
    assert [json.loads(s) for s in result['sampling_params']] == [
        dict(request_id='custom', generation_config=dict(temperature=.5)),
        dict(request_id='custom', generation_config=dict(temperature=.5)),
    ]
    assert result['responses'] == ['first', 'second']
    assert result['scores'] == [1., -1.]
