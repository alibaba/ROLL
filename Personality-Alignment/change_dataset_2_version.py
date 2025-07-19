import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# 加载原始数据集
data = []
with open('/home/qlicv/AReaL/Personality-Alignment/cleaned_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
        # 替换为 Qwen3-8B 模型
        
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
def generate_distractors_batch(prompts, correct_outputs):
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(correct_outputs, str):
        correct_outputs = [correct_outputs]
    inputs = [f"{prompt}\n错误答案：" for prompt in prompts]
    input_tokens = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**input_tokens, max_new_tokens=50, num_return_sequences=1)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    distractors = []
    for i, text in enumerate(generated_texts):
        distractor = text.replace(inputs[i], "").strip()
        if distractor and distractor != correct_outputs[i]:
            distractors.append(distractor)
        else:
            distractors.append("无法识别的答案")
    return distractors if len(distractors) > 1 else distractors[0]

# Initialize new data list
new_data = []

# Get total number of items
total_items = len(data)

# Process data with proper progress tracking
for i, item in enumerate(data, 1):
    # Update progress bar
    print(f"Processing item {i}/{total_items} ({(i/total_items)*100:.1f}%)", end='\r')
    
    qid = item['qid']
    prompt = item['prompt']
    correct_output = item['output']
    
    # Generate confusing wrong answer
    distractor = generate_distractors_batch(prompt + "/no_think", correct_output)

    # Randomly decide if correct answer is A or B
    options = ['A', 'B']
    random.shuffle(options)
    correct_option = options[0]
    wrong_option = options[1]

    if correct_option == 'A':
        prompt_new = f"{prompt}\nNow please choose the most possible output A or B\nA. {correct_output}\nB. {distractor}"
        output_new = 'A'
    else:
        prompt_new = f"{prompt}\nNow please choose the most possible output A or B\nA. {distractor}\nB. {correct_output}"
        output_new = 'B'

    new_data.append({
        'qid': qid,
        'prompt': prompt_new,
        'output': output_new
    })

# 保存新数据集
with open('/home/qlicv/AReaL/Personality-Alignment/2_changed_dataset.jsonl', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)