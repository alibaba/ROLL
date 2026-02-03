import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from roll.utils.checkpoint_manager import download_model

if False:
    from sglang.srt.entrypoints.engine import Engine
else:
    from roll.third_party.sglang import patch as sglang_patch
    Engine = sglang_patch.engine.engine_module.Engine


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def main():
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    model_path = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    model_path = download_model(model_path)

    model = Engine(
        model_path=model_path,
        skip_tokenizer_init=False,
        trust_remote_code=True,
        tp_size=1,
        load_format="auto",
        disable_cuda_graph=False,
        disable_custom_all_reduce=True,
        sampling_backend="pytorch", 
        mem_fraction_static=0.6,
        max_total_tokens=2048,
        max_running_requests=2,
        enable_memory_saver=True,
        quantization="fp8",
        json_model_override_args=
          json.dumps({
            "quantization_config":
            {
              "activation_scheme": "dynamic",
              "weight_block_size": [128, 128],
            }
          }),
    )

    prompts = ["类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案"]
    prompts = [chat_format(prompt) for prompt in prompts]

    sampling_params = {
        'min_new_tokens': 128,
        'max_new_tokens': 128,
    }

    output = model.generate(prompt=prompts, sampling_params=sampling_params)
    print(output)

    train_model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
    for name, param in tqdm(iterable=train_model.named_parameters()):
        model.update_weights_from_tensor(named_tensors=[(name, param)])

    output = model.generate(prompt=prompts, sampling_params=sampling_params)
    print(output)

if __name__ == "__main__":
    main()
