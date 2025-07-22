"""
LLM Generater
"""

import json
import openai
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. LoRA functionality will be disabled.")



class QwenGenerator:
    """Class for evaluating dialogue outputs using Qwen"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto", inference_batch_size: int = 8, lora_path: str = None):
        """
        Initialize Qwen evaluator
        
        Args:
            model_path: Path to Qwen model or HuggingFace model name
            device: Device to use for inference ('auto', 'cuda', 'cpu')
            inference_batch_size: Batch size for model inference
            lora_path: Path to LoRA adapter model (optional)
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.inference_batch_size = inference_batch_size
        
        logger.info(f"Loading Qwen model from: {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        # Load LoRA adapter if specified
        if self.lora_path and PEFT_AVAILABLE:
            logger.info(f"Loading LoRA adapter from: {self.lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            logger.info("LoRA adapter loaded successfully")
        elif self.lora_path and not PEFT_AVAILABLE:
            logger.error("LoRA path specified but PEFT library is not available. Please install peft library.")
            raise ImportError("PEFT library is required for LoRA functionality. Install with: pip install peft")
        
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info("Qwen model loaded successfully")
    
    def call_qwen_batch(self, prompts: List[str]) -> List[Optional[str]]:
        """Call Qwen model for batch inference"""
        try:
            # Format prompts for chat
            formatted_texts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                formatted_texts.append(text)
            
            # Tokenize inputs with padding
            model_inputs = self.tokenizer(
                formatted_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=4096
            ).to(self.model.device)
            
            # Generate responses
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode responses
            input_lengths = model_inputs.input_ids.shape[1]
            generated_ids = generated_ids[:, input_lengths:]
            
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return [response.strip() if response else None for response in responses]
            
        except Exception as e:
            logger.error(f"Qwen batch inference failed: {e}")
            return [None] * len(prompts)
    
    def call_qwen(self, prompt: str) -> Optional[str]:
        """Call Qwen model for single inference"""
        results = self.call_qwen_batch([prompt])
        return results[0] if results else None
    
    def call_qwen_batchly(self, prompts_raw: List[Dict[str, str]]) -> List[Optional[str]]:
        """Call Qwen model for batch inference with progress bar"""
        prompts = [item['prompt'] for item in prompts_raw]
        results = []
        with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
            for i in range(0, len(prompts), self.inference_batch_size):
                batch = prompts[i:i + self.inference_batch_size]
                batch_results = self.call_qwen_batch(batch)
                results.extend([{'qid': prompts_raw[i + j]['id'], 'response': batch_results[j]} for j in range(len(batch_results))])
                pbar.update(len(batch))
        return results
    
def load_prompts(file_path: str, max_limit: int) -> List[Dict[str, str]]:
    """load prompts from a JSONL file"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(prompts) >= max_limit:
                break
            data = json.loads(line.strip())
            prompts.append({'id': data['qid'], 'prompt': data['prompt']})
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Evaluate dialogue outputs using Qwen model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path to Qwen model or HuggingFace model name")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter model (optional)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for inference ('auto', 'cuda', 'cpu')")
    parser.add_argument("--inference_batch_size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to the JSONL file containing prompts")
    parser.add_argument("--output_file", type=str, default="qwen_results.jsonl", help="File to save the results")
    parser.add_argument("--max_limit", type=int, default=1000, help="Maximum number of prompts to process")
    args = parser.parse_args()
    
    # Initialize Qwen generator
    generator = QwenGenerator(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
        inference_batch_size=args.inference_batch_size
    )
    logger.info(f"Using prompts file: {args.prompts_file}")
    # Load prompts
    prompts = load_prompts(args.prompts_file, args.max_limit)
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    if not prompts:
        logger.warning("No prompts found to process.")
        return
    
    # Process prompts in batches
    results = generator.call_qwen_batchly(prompts)
    logger.info("Processing completed")
    # Save results to output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Results saved to {args.output_file}")
    
if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
