from datasets import Dataset
from roll.pipeline.distill.distill_config import DistillConfig
from roll.configs.worker_config import WorkerConfig
from roll.configs.data_args import DataArguments
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.distill.distill_pipeline import preprocess_dataset


def test_preprocess_dataset_with_real_data():
    # ===== 1. 构造两条真实数据 =====
    data = [
        {
            "question_zh": "Natalia在四月份向她的48个朋友出售了夹子，然后在五月份卖出了四月份的一半。Natalia在四月和五月总共卖了多少个夹子？",
            "answer_zh": "Natalia在五月份卖出了48/2 = 24个夹子。\nNatalia在四月和五月总共卖出了48+24 = 72个夹子。"
        },
        {
            "question_zh": "翁做保姆工作每小时赚12美元。昨天，她只做了50分钟的保姆工作。她赚了多少钱？",
            "answer_zh": "翁每分钟赚12/60 = 0.2美元。\n工作了50分钟，她赚了0.2 x 50 = 10美元。\n答案是：10。",
        }
    ]
    dataset = Dataset.from_list(data)

    # ===== 2. 创建DistillConfig对象 =====
    local_or_mirror_model_path = "Qwen/Qwen2.5-0.5B-Instruct"

    student_cfg = WorkerConfig(data_args=DataArguments(preprocessing_num_workers=16))
    student_cfg.model_args.model_name_or_path = local_or_mirror_model_path

    teacher_cfg = WorkerConfig(data_args=DataArguments(preprocessing_num_workers=16))
    teacher_cfg.model_args.model_name_or_path = local_or_mirror_model_path

    pipeline_config = DistillConfig(
        student=student_cfg,
        teacher=teacher_cfg,
        query_key="question_zh",
        response_key="answer_zh",
        distill_on_prompt=True,
        sequence_length=256
    )

    # ===== 3. 加载tokenizer =====
    tokenizer = default_tokenizer_provider(model_args=pipeline_config.student.model_args)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== 5. 跑 preprocess_dataset 全流程 =====
    processed_ds = preprocess_dataset(dataset, tokenizer, pipeline_config)

    # ===== 6. 打印处理结果 =====
    print("\n=== 处理后数据（含文本） ===")
    for i in range(len(processed_ds)):
        item = processed_ds[i]
        print(f"\n----- 样本 {i} -----")

        # input_ids -> 原始输入文本
        input_text = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
        print("原始输入文本:")
        print(input_text)

        # 去掉 labels 中=-100的位置
        label_ids = [tid for tid in item["labels"] if tid != -100]
        label_text = tokenizer.decode(label_ids, skip_special_tokens=False)
        print("监督文本（仅真实监督部分）:")
        print(label_text)

        # 原始 token 和 label 数字
        print("input_ids:", item["input_ids"])
        print("labels   :", item["labels"])


if __name__ == "__main__":
    test_preprocess_dataset_with_real_data()
