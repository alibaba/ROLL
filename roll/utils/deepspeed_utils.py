from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
"""
ref: https://github.com/OpenRLHF/OpenRLHF/blob/494850f50342ed38d5ae76ef45a3207f3523b582/openrlhf/utils/deepspeed/deepspeed_utils.py#L104
"""

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    """
    用于生成优化器的参数组列表，参数分组，为了在训练时可以对不同的参数应用不同的权重衰减策略
    Args:
        model:
        weight_decay:
        no_decay_name_list:

    Returns:

    """
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return [group for group in optimizer_grouped_parameters if group["params"]]


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]
