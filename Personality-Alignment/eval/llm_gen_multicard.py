"""
LLM Generator with Multi-Card Data Parallel Support
支持多卡数据并行推理的LLM生成器
"""

import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import os
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np
from tqdm import tqdm
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from peft import PeftModel, PeftConfig

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. LoRA functionality will be disabled.")


class MultiCardQwenGenerator:
    """Multi-card data parallel Qwen generator"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_path: str = None,
        inference_batch_size: int = 8,
        num_gpus: int = None,
        per_gpu_batch_size: int = None,
    ):
        """
        Initialize Multi-card Qwen generator

        Args:
            model_path: Path to Qwen model or HuggingFace model name
            lora_path: Path to LoRA adapter model (optional)
            inference_batch_size: Total batch size across all GPUs
            num_gpus: Number of GPUs to use (auto-detect if None)
            per_gpu_batch_size: Batch size per GPU (auto-calculate if None)
        """
        self.model_path = model_path
        self.lora_path = lora_path

        # GPU configuration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        self.num_gpus = min(self.num_gpus, torch.cuda.device_count())

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available")

        logger.info(f"Using {self.num_gpus} GPUs for inference")

        # Batch size configuration
        if per_gpu_batch_size is not None:
            self.per_gpu_batch_size = per_gpu_batch_size
            self.total_batch_size = per_gpu_batch_size * self.num_gpus
            logger.info(
                f"Using per-GPU batch size: {self.per_gpu_batch_size}, total batch size: {self.total_batch_size}"
            )
        else:
            self.total_batch_size = inference_batch_size
            self.per_gpu_batch_size = max(1, inference_batch_size // self.num_gpus)
            logger.info(
                f"Using total batch size: {self.total_batch_size}, per-GPU batch size: {self.per_gpu_batch_size}"
            )

        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading tokenizer from: {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model from: {self.model_path}")

        # Load base model on first GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map=None,  # We'll handle device placement manually
        )

        # Load LoRA adapter if specified
        if self.lora_path and PEFT_AVAILABLE:
            logger.info(f"Loading LoRA adapter from: {self.lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            logger.info("LoRA adapter loaded successfully")
        elif self.lora_path and not PEFT_AVAILABLE:
            logger.error("LoRA path specified but PEFT library is not available")
            raise ImportError("PEFT library is required for LoRA functionality")

        # Move model to first GPU and wrap with DataParallel if using multiple GPUs
        self.model = self.model.cuda(0)

        if self.num_gpus > 1:
            device_ids = list(range(self.num_gpus))
            self.model = DataParallel(self.model, device_ids=device_ids)
            logger.info(f"Model wrapped with DataParallel across GPUs: {device_ids}")

        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.model.eval()
        logger.info("Model loaded and configured successfully")

    def call_qwen_batch(self, prompts: List[str]) -> List[Optional[str]]:
        """Call Qwen model for batch inference with automatic batching"""
        if not prompts:
            return []

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

            # Determine optimal batch size for current input
            effective_batch_size = min(len(prompts), self.total_batch_size)

            # Process in chunks if needed
            all_responses = []
            for i in range(0, len(formatted_texts), effective_batch_size):
                batch_texts = formatted_texts[i : i + effective_batch_size]
                batch_responses = self._process_batch(batch_texts)
                all_responses.extend(batch_responses)

            return all_responses

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [None] * len(prompts)

    def _process_batch(self, batch_texts: List[str]) -> List[Optional[str]]:
        """Process a single batch of texts"""
        try:
            # Tokenize inputs with padding
            model_inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
            )

            # Move inputs to GPU
            input_ids = model_inputs.input_ids.cuda(0)
            attention_mask = model_inputs.attention_mask.cuda(0)

            # Generate responses
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode responses
            input_lengths = input_ids.shape[1]
            generated_ids = generated_ids[:, input_lengths:]

            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return [response.strip() if response else None for response in responses]

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM in batch processing. Batch size: {len(batch_texts)}")
            # Try to process with smaller batches
            if len(batch_texts) > 1:
                logger.info("Retrying with smaller batches...")
                mid = len(batch_texts) // 2
                batch1 = self._process_batch(batch_texts[:mid])
                batch2 = self._process_batch(batch_texts[mid:])
                return batch1 + batch2
            else:
                logger.error("OOM with batch size 1, skipping this sample")
                return [None]
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [None] * len(batch_texts)

    def call_qwen_batchly(self, prompts_raw: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process prompts in optimized batches with progress tracking"""
        prompts = [item["prompt"] for item in prompts_raw]

        results = []
        with tqdm(total=len(prompts), desc=f"Processing prompts (GPUs: {self.num_gpus})") as pbar:
            for i in range(0, len(prompts), self.total_batch_size):
                batch_prompts = prompts[i : i + self.total_batch_size]
                batch_responses = self.call_qwen_batch(batch_prompts)

                # Format results
                for j, response in enumerate(batch_responses):
                    results.append({"qid": prompts_raw[i + j]["id"], "response": response})

                pbar.update(len(batch_prompts))

                # Optional: Clear cache periodically
                if (i // self.total_batch_size) % 10 == 0:
                    torch.cuda.empty_cache()

        return results

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        memory_info = {}
        for i in range(self.num_gpus):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            memory_info[f"gpu_{i}"] = {"allocated_gb": memory_allocated, "reserved_gb": memory_reserved}
        return memory_info


class AsyncMultiCardQwenGenerator:
    """Async version with pipeline parallelism for even better throughput"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_path: str = None,
        inference_batch_size: int = 8,
        num_gpus: int = None,
        queue_size: int = 100,
    ):
        """
        Initialize Async Multi-card Qwen generator with pipeline parallelism

        Args:
            model_path: Path to Qwen model or HuggingFace model name
            lora_path: Path to LoRA adapter model (optional)
            inference_batch_size: Batch size per GPU
            num_gpus: Number of GPUs to use
            queue_size: Size of the processing queue
        """
        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        self.num_gpus = min(self.num_gpus, torch.cuda.device_count())

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available")

        logger.info(f"Using {self.num_gpus} GPUs for async inference")

        self.model_path = model_path
        self.lora_path = lora_path
        self.inference_batch_size = inference_batch_size
        self.queue_size = queue_size

        # Create separate generators for each GPU
        self.generators = []
        self._load_models()

    def _load_models(self):
        """Load separate model instances for each GPU"""
        for gpu_id in range(self.num_gpus):
            logger.info(f"Loading model on GPU {gpu_id}")

            # Create generator for this GPU
            generator = self._create_single_gpu_generator(gpu_id)
            self.generators.append(generator)

        logger.info(f"All {self.num_gpus} model instances loaded successfully")

    def _create_single_gpu_generator(self, gpu_id: int):
        """Create a single GPU generator"""
        # Set device
        device = f"cuda:{gpu_id}"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model on specific GPU
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map=None,
        )

        # Load LoRA if specified
        if self.lora_path and PEFT_AVAILABLE:
            model = PeftModel.from_pretrained(model, self.lora_path)

        model = model.to(device)
        model.eval()

        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        return {
            "model": model,
            "tokenizer": tokenizer,
            "generation_config": generation_config,
            "device": device,
            "gpu_id": gpu_id,
        }

    def _process_batch_on_gpu(self, batch_texts: List[str], generator: Dict) -> List[Optional[str]]:
        """Process batch on specific GPU"""
        try:
            model = generator["model"]
            tokenizer = generator["tokenizer"]
            generation_config = generator["generation_config"]
            device = generator["device"]

            # Tokenize
            model_inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
            ).to(device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode
            input_lengths = model_inputs.input_ids.shape[1]
            generated_ids = generated_ids[:, input_lengths:]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            return [response.strip() if response else None for response in responses]

        except Exception as e:
            logger.error(f"Error on GPU {generator['gpu_id']}: {e}")
            return [None] * len(batch_texts)

    def call_qwen_async(self, prompts_raw: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process prompts using all GPUs in parallel"""
        prompts = [item["prompt"] for item in prompts_raw]

        # Format prompts
        formatted_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.generators[0]["tokenizer"].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            formatted_texts.append(text)

        # Distribute batches across GPUs
        results = [None] * len(prompts)
        futures = []

        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            batch_idx = 0
            for i in range(0, len(formatted_texts), self.inference_batch_size):
                batch_texts = formatted_texts[i : i + self.inference_batch_size]
                gpu_id = batch_idx % self.num_gpus
                generator = self.generators[gpu_id]

                future = executor.submit(self._process_batch_on_gpu, batch_texts, generator)
                futures.append((future, i, len(batch_texts)))
                batch_idx += 1

            # Collect results with progress bar
            with tqdm(total=len(futures), desc=f"Processing batches on {self.num_gpus} GPUs") as pbar:
                for future, start_idx, batch_size in futures:
                    batch_responses = future.result()

                    # Store results
                    for j, response in enumerate(batch_responses):
                        if start_idx + j < len(results):
                            results[start_idx + j] = {"qid": prompts_raw[start_idx + j]["id"], "response": response}

                    pbar.update(1)

        # Filter out None results
        return [r for r in results if r is not None]


def load_prompts(file_path: str, max_limit: int) -> List[Dict[str, str]]:
    """Load prompts from a JSONL file"""
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_limit:
                break
            data = json.loads(line.strip())
            prompts.append({"id": data["qid"], "prompt": data["prompt"]})
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Multi-card Qwen inference")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path to Qwen model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--inference_batch_size", type=int, default=32, help="Total batch size across all GPUs")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=None, help="Batch size per GPU (overrides total batch size)"
    )
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (auto-detect if None)")
    parser.add_argument("--use_async", action="store_true", help="Use async version for better parallelism")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to JSONL file with prompts")
    parser.add_argument("--output_file", type=str, default="qwen_multicard_results.jsonl", help="Output file path")
    parser.add_argument("--max_limit", type=int, default=1000, help="Maximum number of prompts to process")

    args = parser.parse_args()

    try:
        # Load prompts
        prompts = load_prompts(args.prompts_file, args.max_limit)
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

        if not prompts:
            logger.warning("No prompts found to process.")
            return

        # Initialize generator
        if args.use_async:
            logger.info("Using async multi-card generator")
            generator = AsyncMultiCardQwenGenerator(
                model_path=args.model_path,
                lora_path=args.lora_path,
                inference_batch_size=args.per_gpu_batch_size or args.inference_batch_size,
                num_gpus=args.num_gpus,
            )
            results = generator.call_qwen_async(prompts)
        else:
            logger.info("Using synchronized multi-card generator")
            generator = MultiCardQwenGenerator(
                model_path=args.model_path,
                lora_path=args.lora_path,
                inference_batch_size=args.inference_batch_size,
                num_gpus=args.num_gpus,
                per_gpu_batch_size=args.per_gpu_batch_size,
            )

            # Log memory usage
            memory_info = generator.get_memory_usage()
            logger.info(f"GPU memory usage: {memory_info}")

            results = generator.call_qwen_batchly(prompts)

        logger.info("Processing completed")

        # Save results
        with open(args.output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"Results saved to {args.output_file}")
        logger.info(f"Processed {len(results)} prompts successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        # Clear GPU memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
