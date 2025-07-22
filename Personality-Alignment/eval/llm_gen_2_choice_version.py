from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.distributed as dist
import json
import random
import pandas as pd
from tqdm import tqdm

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
def load_model_and_tokenizer():
    # Load model and tokenizer
    model_name = "Qwen/Qwen3-8B"  # Adjust the model name as needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    return model, tokenizer

def batch_generate_responses(model, tokenizer, prompts, batch_size=32):
        predictions = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches", total=num_batches):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=0.7
                )
            
            batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, response in zip(batch_prompts, batch_responses):
                response = response[len(prompt):].strip().upper()
                if 'A' in response:
                    predictions.append('A')
                elif 'B' in response:
                    predictions.append('B')
                else:
                    predictions.append(None)
                    
        return predictions

def evaluate_model(data_path):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
    
    results = []
    correct = 0
    total = 0
    
    # Process each example
    for item in tqdm(data):
        qid = item['qid']
        prompt = item['prompt']
        ground_truth = item['output']
        
        # Get model's prediction
        predictions = batch_generate_responses(model, tokenizer, [prompt])
        prediction = predictions[0] if predictions else None
        
        if prediction is not None:
            total += 1
            if prediction == ground_truth:
                correct += 1
                
        results.append({
            'qid': qid,
            'prompt': prompt,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': prediction == ground_truth if prediction else False
        })
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    try:
        # Specify your dataset path
        data_path = "/home/zyangdm/ROLL/Personality-Alignment/3_changed_dialogue_dataset_nothink.jsonl"
        results_df, final_accuracy = evaluate_model(data_path)
        print(f"Final accuracy: {final_accuracy:.2%}")
        # Optionally save results
        results_df.to_csv("evaluation_results.csv", index=False)
    finally:
        cleanup()
        evaluate_model(data_path)
    data_path = "/home/zyangdm/ROLL/Personality-Alignment/3_changed_dialogue_dataset_nothink.jsonl"
    results_df, final_accuracy = evaluate_model(data_path)
    
    # Optionally save results
    results_df.to_csv("evaluation_results.csv", index=False)