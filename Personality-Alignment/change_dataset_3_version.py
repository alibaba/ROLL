import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# 加载原始数据集
data = []
with open('/home/qlicv/AReaL/Personality-Alignment/dialogue_dataset_all_v2_cleaned.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

def load_qwen3_8b(model_path="Qwen/Qwen3-8B", device_map="auto", quantize=False):
    """
    加载 Qwen3-8B 模型和分词器

    参数:
    model_path: 模型路径 (Hugging Face ID 或本地路径)
    device_map: 设备映射 ("auto", "cuda", "cpu")
    quantize: 是否使用 4-bit 量化 (减少显存需求)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    return model, tokenizer

# 初始化LLM用于生成混淆性错误答案
model_path = "Qwen/Qwen3-8B"
model, tokenizer = load_qwen3_8b(model_path=model_path, device_map="auto", quantize=False)

def generate_distractors_batch(prompts, correct_outputs, batch_size=8):
    """
    真正的批量生成干扰项
    
    参数:
    prompts: 提示词列表
    correct_outputs: 正确答案列表
    batch_size: 批量大小
    """
    all_distractors = []
    
    # 准备所有输入
    all_inputs = [f"{prompt}\n错误答案：" for prompt in prompts]
    
    # 分批处理
    for i in tqdm(range(0, len(all_inputs), batch_size), desc="Generating distractors"):
        batch_inputs = all_inputs[i:i+batch_size]
        batch_correct = correct_outputs[i:i+batch_size]
        
        # 批量tokenize
        input_tokens = tokenizer(
            batch_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(model.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens, 
                max_new_tokens=50, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 批量解码
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 处理这一批的结果
        batch_distractors = []
        for j, text in enumerate(generated_texts):
            distractor = text.replace(batch_inputs[j], "").strip()
            if distractor and distractor != batch_correct[j]:
                batch_distractors.append(distractor)
            else:
                batch_distractors.append("无法识别的答案")
        
        all_distractors.extend(batch_distractors)
    
    return all_distractors

def process_data_batch(data, batch_size=8):
    """
    批量处理数据
    """
    # 提取所有需要的信息
    prompts = [item['prompt'] + "/no_think" for item in data]
    correct_outputs = [item['output'] for item in data]
    qids = [item['qid'] for item in data]
    
    print("开始批量生成干扰项...")
    # 批量生成所有干扰项
    distractors = generate_distractors_batch(prompts, correct_outputs, batch_size)
    
    print("构建新数据集...")
    # 构建新数据集
    new_data = []
    for i, (qid, original_prompt, correct_output, distractor) in enumerate(
        zip(qids, [item['prompt'] for item in data], correct_outputs, distractors)
    ):
        # 随机决定正确答案是A还是B
        options = ['A', 'B']
        random.shuffle(options)
        correct_option = options[0]
        
        if correct_option == 'A':
            prompt_new = f"{original_prompt}\nNow please choose the most possible output A or B\nA. {correct_output}\nB. {distractor}"
            output_new = 'A'
        else:
            prompt_new = f"{original_prompt}\nNow please choose the most possible output A or B\nA. {distractor}\nB. {correct_output}"
            output_new = 'B'
        
        new_data.append({
            'qid': qid,
            'prompt': prompt_new,
            'output': output_new
        })
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(data)} 条数据")
    
    return new_data

# 设置批量大小（根据显存调整）
BATCH_SIZE = 32  # 可以根据显存大小调整

# 批量处理数据
print(f"开始处理 {len(data)} 条数据，批量大小: {BATCH_SIZE}")
new_data = process_data_batch(data, batch_size=BATCH_SIZE)

# 保存新数据集
print("保存新数据集...")
with open('/home/qlicv/AReaL/Personality-Alignment/3_changed_dataset.jsonl', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"处理完成！共生成 {len(new_data)} 条新数据")