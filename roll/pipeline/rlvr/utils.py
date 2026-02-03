import os.path
import json
import time
import numpy
import copy
import requests

import torch

from codetiming import Timer
import multiprocessing

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger


logger = get_logger()

COLUMMNS_CONFIG = [
        ['global_step','bigint'],
        ['id','string'],
        ['source','string'],
        ['difficulty','string'],
        ['prompt','string'],
        ['messages','string'],
        ['ground_truth','string'],
        ['case_type','string'],
        ['test_case_function','string'],
        ['test_cases','string'],
        ['tag','string'],
        ['domain','string'],
        ['responses','string'],
        ['scores','double'],
        ['sampling_params','string']
    ]

def write_to_json_process(path, data, columns_configs):
    os.makedirs(path, exist_ok=True)
    column_names = {item[0] for item in columns_configs}
    data = {k: v.tolist() if isinstance(v, numpy.ndarray) else v for k,v in data.items() if k in column_names}
    with Timer(name="dump", logger=None) as timer:
        global_step = data.get('global_step', [0])[0]
        with open(os.path.join(path, f"rollout_dump_data.step_{global_step}.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logger.info(f"dump_rollout to {path}: {timer.last}")

def json_checker(path:str):
    return path.startswith("/")

DUMPING_FUNC = [
    [json_checker, write_to_json_process],
]


def dump_rollout_to_specific_path(path: str, global_step: int, data: DataProto, tokenizer):
    if not path:
        return
    write_data = copy.deepcopy(data.non_tensor_batch)
    responses = tokenizer.batch_decode(data.batch['responses'], skip_special_tokens=True)
    data_cnt = len(responses)
    write_data['responses'] = responses
    scores = data.batch['scores'].tolist()
    write_data['scores'] = scores
    meta_info = [json.dumps(data.meta_info)] * data_cnt
    write_data['sampling_params'] = meta_info
    write_data['global_step'] = [global_step] * data_cnt

    # TODO:If IO becomes the bottleneck, need use queue and only one write process to dump data
    for checker, func in DUMPING_FUNC:
        if checker(path):
            p = multiprocessing.Process(target=func, args=(path, write_data, COLUMMNS_CONFIG), daemon=True)
            p.start()
