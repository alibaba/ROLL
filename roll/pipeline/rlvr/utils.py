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
    write_data = {k: copy.deepcopy(v) for k, v in data.non_tensor_batch.items()}
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

def dump_batch_to_reward_system(data: DataProto, tokenizer):
    """

    Args:
        reward_system_config (dict): _description_
        global_step (int): _description_
        data (DataProto): _description_
        tokenizer (_type_): _description_
    """
    def write_log_to_reward_system(dump_item, reward_system_config, max_retries=3, timeout=120):

        write_log_url = reward_system_config.get(
            "write_log_url", 
            "http://xrl.alibaba-inc.com/apis/custom/reward-system/v1/write_log"
            )
        xrl_authorization = reward_system_config.get("xrl_authorization", "t-29imkiykio27pmju")
        reward_system_type = reward_system_config.get("reward_system_type", "online")
        
        headers = {
            "Content-Type": "application/json",
            "XRL-Authorization": f"Bearer {xrl_authorization}",
            "BusinessType": reward_system_type
            }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    write_log_url,
                    headers=headers,
                    json=dump_item,
                    timeout=timeout
                )
                response.raise_for_status()  # 如果状态码不是 2xx，会抛出 HTTPError
                response_json = response.json()
                errcode = int(response_json.get("errcode", -1))
                
                if errcode == 0:
                    break
                else:
                    errmsg = response_json.get("errmsg", "unknown error")
                    logger.error(f"Reward system log failed; Try {attempt} out of {max_retries} times,{errcode=}, {errmsg=}")

            except Exception as e:
                logger.error(f"Reward system log failed; Try {attempt} out of {max_retries} times, Error: {e}")
    
    def dump_to_reward_system(reward_system_config, dump_data):
        
        project_id = reward_system_config.get("project_id", 'roll_default_experiment_id')
        experiment_id = reward_system_config.get("experiment_id", 'roll_default_task_id')

        for idx in range(len(dump_data["responses_ids"])):
            dump_item = {
                "project_id": project_id,
                "experiment_id": experiment_id
            }

            seq_len = len(dump_data["responses_ids"][idx])
            
            keys_to_truncate = [
                "log_prob",
                "old_log_prob",
                "ref_log_prob",
                "token_level_reward",
                "values",
                "adv",
                "response_token_str_list",
            ]

            for key in dump_data.keys():
                if (
                    key in keys_to_truncate
                    and isinstance(dump_data[key][idx], list)
                    and len(dump_data[key][idx]) > seq_len
                ):
                    dump_item[key] = dump_data[key][idx][:seq_len]
                else:
                    dump_item[key] = dump_data[key][idx]
            
            write_log_to_reward_system(dump_item, reward_system_config)
    
    def remove_leading_zeros(A, r_mask):
        B = []
        for i in range(len(A)):
            row = A[i]
            mask = (r_mask[i] != 0).to(torch.int32)
            if not mask.any():  # 如果该行全为零
                B.append([])  # 添加空列表
            else:
                first_non_zero = mask.argmax().item()  # 找到第一个非零元素的索引
                B.append(row[first_non_zero:].tolist())
        return B
    
    def remove_eos_and_pad_token_id(token_ids_batch, tokenizer):
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        if eos_id is None:
            eos_id = tokenizer.sep_token_id

        cleaned = []
        for seq in token_ids_batch:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            filtered = [tid for tid in seq if tid != eos_id and tid != pad_id]
            cleaned.append(filtered)
        return cleaned
    
    
    reward_system_config = data.meta_info.get("reward_system_config", {})
    global_step = data.meta_info.get("global_step", 0)
    if not reward_system_config.get("project_id", None):
        return
    
    responses_ids = remove_eos_and_pad_token_id(data.batch["responses"], tokenizer)
    response_token_str_list = [
        tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        for ids in data.batch["responses"]
    ]

    data_cnt = len(responses_ids)
    
    ref_log_probs = remove_leading_zeros(
            data.batch["ref_log_probs"], data.batch["response_mask"][:, 1:]
        )
    old_log_probs = remove_leading_zeros(
            data.batch["old_log_probs"], data.batch["response_mask"][:, 1:]
        )
    advantages = remove_leading_zeros(
            data.batch["advantages"], data.batch["response_mask"][:, 1:]
        )
    token_level_rewards = remove_leading_zeros(
            data.batch["token_level_rewards"], data.batch["response_mask"][:, 1:]
        )

    if "values" in data.batch:
        values = data.batch["values"].float()
    else:
        values = [0] * data_cnt
    
    # TODO: get log_probs
    log_probs = [0] * data_cnt
    
    non_tensor_batch = copy.deepcopy(data.non_tensor_batch)
    
    dump_data = {
        "data_id": non_tensor_batch.get('id', [0] * data_cnt),
        "rollout_id": non_tensor_batch.get('rollout_id', [0] * data_cnt),
        "global_step": [global_step] * data_cnt,
        "responses_ids": responses_ids,
        "log_prob": log_probs,
        "old_log_prob": old_log_probs,
        "ref_log_prob": ref_log_probs,
        "token_level_reward": token_level_rewards,
        "values": values,
        "adv": advantages,
        "response_token_str_list": response_token_str_list
    }
    
    p = multiprocessing.Process(
        target=dump_to_reward_system, 
        args=(reward_system_config, dump_data), daemon=True
        )
    p.start()