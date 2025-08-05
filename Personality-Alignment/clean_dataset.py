"""
现在的dialog中output字段里面包含了很多祈使句or问题的形式，这样可能导致model没有办法去simulate真正的对话。
（此外，似乎还有一些typo的问题，这些应该都会影响model的表现，需要清洗）

现在可能有几种做法：
1. 直接过滤掉这些output，形成一个纯的对话数据集。
2. 对这些output进行改写，变成陈述句的形式。
3. 修改这个对话数据集为选择题的形式，给出几个选项让model选择。
"""

import json
import openai
import time
import argparse
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPTDatasetCleaner:
    """Class for dataset cleaning using GPT"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize GPT cleaner
        
        Args:
            api_key: OpenAI API key, if None will get from OPENAI_API_KEY environment variable
            base_url: API base URL, supports custom API services
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        
        if not self.api_key:
            raise ValueError("API key not found, please set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Cleaning prompt templates
        self.cleaning_prompts = {
            "typo_fix": """Please fix spelling errors, grammar mistakes, and formatting issues in the following text while preserving the original meaning:

Original text: {text}

Requirements:
1. Fix obvious spelling errors
2. Correct grammar issues
3. Standardize punctuation usage
4. Maintain the original semantics and tone
5. Return only the corrected text without any explanations

Corrected text:""",

            "convert_to_statement": """Please convert the following imperative sentences or questions into natural declarative statements while preserving the original meaning:

Original text: {text}

Requirements:
1. Convert imperative sentences to declarative statements
2. Convert questions to declarative statements
3. Maintain the core information of the original text
4. Make the statements natural and fluent
5. Return only the converted text without any explanations

Converted text:""",

            "judge_quality": """Please judge whether the following text is suitable as a response in a dialogue dataset:

Text: {text}

Please evaluate from the following dimensions:
1. Language quality: Are there serious grammar or spelling errors?
2. Content appropriateness: Does it contain inappropriate content?
3. Dialogue relevance: Is it suitable as a dialogue response?
4. Information completeness: Is it expressed completely?

Please answer only "KEEP" or "FILTER":""",

            "comprehensive_clean": """Please comprehensively clean the following dialogue text:

Original text: {text}

Tasks:
1. Fix all spelling errors and grammar issues
2. Convert imperative sentences and questions to appropriate declarative statements
3. Standardize punctuation and formatting
4. Ensure the text is suitable as a dialogue dataset response
5. Maintain the core information and tone of the original text

If the text quality is too poor to be fixed, please reply "DELETE"
Otherwise, return only the cleaned text without any explanations:"""
        }
    
    def call_gpt(self, prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[str]:
        """Call GPT API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
            
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit reached, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("Maximum retry attempts reached, skipping this entry")
                    return None
            
            except Exception as e:
                logger.error(f"GPT API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return None
        
        return None
    
    def fix_typos_with_gpt(self, text: str) -> Optional[str]:
        """Fix typos using GPT"""
        prompt = self.cleaning_prompts["typo_fix"].format(text=text)
        return self.call_gpt(prompt)
    
    def convert_to_statement_with_gpt(self, text: str) -> Optional[str]:
        """Convert imperatives/questions to statements using GPT"""
        prompt = self.cleaning_prompts["convert_to_statement"].format(text=text)
        return self.call_gpt(prompt)
    
    def judge_quality_with_gpt(self, text: str) -> bool:
        """Use GPT to judge text quality"""
        prompt = self.cleaning_prompts["judge_quality"].format(text=text)
        result = self.call_gpt(prompt)
        return result and "KEEP" in result.upper()
    
    def comprehensive_clean_with_gpt(self, text: str) -> Optional[str]:
        """Comprehensive cleaning using GPT"""
        prompt = self.cleaning_prompts["comprehensive_clean"].format(text=text)
        result = self.call_gpt(prompt)
        
        if result and result.upper() == "DELETE":
            return None
        return result
    
    def clean_single_entry(self, entry: Dict[str, Any], strategy: str = "comprehensive") -> Optional[Dict[str, Any]]:
        """Clean a single data entry"""
        if 'output' not in entry:
            return entry
        
        output_text = entry['output']
        if not isinstance(output_text, str) or not output_text.strip():
            return entry
        
        cleaned_entry = entry.copy()
        
        try:
            if strategy == "typo_only":
                # Only fix typos
                cleaned_text = self.fix_typos_with_gpt(output_text)
            elif strategy == "convert_only":
                # Only convert sentence types
                cleaned_text = self.convert_to_statement_with_gpt(output_text)
            elif strategy == "filter":
                # Judge quality and filter
                if not self.judge_quality_with_gpt(output_text):
                    return None
                cleaned_text = output_text
            elif strategy == "comprehensive":
                # Comprehensive cleaning
                cleaned_text = self.comprehensive_clean_with_gpt(output_text)
            else:
                raise ValueError(f"Unknown cleaning strategy: {strategy}")
            
            if cleaned_text is None:
                return None
            
            cleaned_entry['cleaned_output'] = cleaned_text
            return cleaned_entry
            
        except Exception as e:
            logger.error(f"Error cleaning entry: {e}")
            return entry  # Return original entry on error
    
    def clean_dataset_batch(self, entries: List[Dict[str, Any]], strategy: str = "comprehensive") -> List[Dict[str, Any]]:
        """Clean data entries in batches"""
        cleaned_entries = []
        
        for entry in entries:
            cleaned_entry = self.clean_single_entry(entry, strategy)
            if cleaned_entry is not None:
                cleaned_entries.append(cleaned_entry)
            
            # Add small delay to avoid API limits
            time.sleep(0.1)
        
        return cleaned_entries
    
    def clean_dataset_parallel(self, entries: List[Dict[str, Any]], strategy: str = "comprehensive", 
                             max_workers: int = 5) -> List[Dict[str, Any]]:
        """Clean data entries in parallel"""
        cleaned_entries = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_entry = {
                executor.submit(self.clean_single_entry, entry, strategy): entry 
                for entry in entries
            }
            
            # Collect results
            for future in as_completed(future_to_entry):
                try:
                    result = future.result()
                    if result is not None:
                        cleaned_entries.append(result)
                except Exception as e:
                    logger.error(f"Parallel processing error: {e}")
        
        return cleaned_entries
    
    def load_and_clean_jsonl(self, input_file: str, output_file: str, strategy: str = "comprehensive",
                           parallel: bool = False, max_workers: int = 5, batch_size: int = 100) -> None:
        """Load and clean entire jsonl file"""
        logger.info(f"Starting GPT cleaning for file: {input_file}")
        logger.info(f"Using strategy: {strategy}")
        logger.info(f"Parallel processing: {parallel}")
        
        # Read all data
        entries = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing error at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(entries)} entries")
        
        # Process in batches with global progress bar
        all_cleaned_entries = []
        with tqdm(total=len(entries), desc="Cleaning entries") as pbar:
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(entries) + batch_size - 1)//batch_size}")
                
                if parallel:
                    cleaned_batch = self.clean_dataset_parallel(batch, strategy, max_workers)
                else:
                    cleaned_batch = self.clean_dataset_batch(batch, strategy)
                
                all_cleaned_entries.extend(cleaned_batch)
                
                # Update progress bar
                pbar.update(len(batch))
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_cleaned_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Cleaning completed!")
        logger.info(f"Original entries: {len(entries)}")
        logger.info(f"Cleaned entries: {len(all_cleaned_entries)}")
        logger.info(f"Cleaned file saved to: {output_file}")


class QwenDatasetCleaner:
    """Class for dataset cleaning using Qwen3-8B model"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto", inference_batch_size: int = 8):
        """
        Initialize Qwen cleaner
        
        Args:
            model_path: Path to Qwen model or HuggingFace model name
            device: Device to use for inference ('auto', 'cuda', 'cpu')
            inference_batch_size: Batch size for model inference (adjust based on GPU memory)
        """
        self.model_path = model_path
        self.device = device
        self.inference_batch_size = inference_batch_size
        
        logger.info(f"Loading Qwen model from: {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"  # Ensure left padding for batch processing
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,  # Use deterministic generation for consistency
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info("Qwen model loaded successfully")
        
        # Cleaning prompt templates
        self.cleaning_prompts = {
            "typo_fix": """Please fix spelling errors, grammar mistakes, and formatting issues in the following text while preserving the original meaning:

Original text: {text}

Requirements:
1. Fix obvious spelling errors
2. Correct grammar issues
3. Standardize punctuation usage
4. Maintain the original semantics and tone
5. Return only the corrected text without any explanations

Corrected text:""",

            "convert_to_statement": """Please convert the following imperative sentences or questions into natural declarative statements while preserving the original meaning:

Original text: {text}

Requirements:
1. Convert imperative sentences to declarative statements
2. Convert questions to declarative statements
3. Maintain the core information of the original text
4. Make the statements natural and fluent
5. Return only the converted text without any explanations

Converted text:""",

            "judge_quality": """Please judge whether the following text is suitable as a response in a dialogue dataset:

Text: {text}

Please evaluate from the following dimensions:
1. Language quality: Are there serious grammar or spelling errors?
2. Content appropriateness: Does it contain inappropriate content?
3. Dialogue relevance: Is it suitable as a dialogue response?
4. Information completeness: Is it expressed completely?

Please answer only "KEEP" or "FILTER":""",

            "comprehensive_clean": """Please comprehensively clean the following dialogue text:

Original text: {text}

Tasks:
1. Fix all spelling errors and grammar issues
2. Convert imperative sentences and questions to appropriate declarative statements
3. Standardize punctuation and formatting
4. Ensure the text is suitable as a dialogue dataset response
5. Maintain the core information and tone of the original text

If the text quality is too poor to be fixed, please reply "DELETE"
Otherwise, return only the cleaned text without any explanations:"""
        }
    
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
                max_length=2048
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
            generated_ids = generated_ids[:, input_lengths:]  # Remove input tokens
            
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return [response.strip() if response else None for response in responses]
            
        except Exception as e:
            logger.error(f"Qwen batch inference failed: {e}")
            return [None] * len(prompts)
    
    def call_qwen(self, prompt: str) -> Optional[str]:
        """Call Qwen model for single inference"""
        results = self.call_qwen_batch([prompt])
        return results[0] if results else None
    
    def fix_typos_with_qwen(self, text: str) -> Optional[str]:
        """Fix typos using Qwen"""
        prompt = self.cleaning_prompts["typo_fix"].format(text=text)
        return self.call_qwen(prompt)
    
    def convert_to_statement_with_qwen(self, text: str) -> Optional[str]:
        """Convert imperatives/questions to statements using Qwen"""
        prompt = self.cleaning_prompts["convert_to_statement"].format(text=text)
        return self.call_qwen(prompt)
    
    def judge_quality_with_qwen(self, text: str) -> bool:
        """Use Qwen to judge text quality"""
        prompt = self.cleaning_prompts["judge_quality"].format(text=text)
        result = self.call_qwen(prompt)
        return result and "KEEP" in result.upper()
    
    def comprehensive_clean_with_qwen(self, text: str) -> Optional[str]:
        """Comprehensive cleaning using Qwen"""
        prompt = self.cleaning_prompts["comprehensive_clean"].format(text=text)
        result = self.call_qwen(prompt)
        
        if result and result.upper() == "DELETE":
            return None
        return result
    
    def clean_single_entry(self, entry: Dict[str, Any], strategy: str = "comprehensive") -> Optional[Dict[str, Any]]:
        """Clean a single data entry"""
        if 'output' not in entry:
            return entry
        
        output_text = entry['output']
        if not isinstance(output_text, str) or not output_text.strip():
            return entry
        
        cleaned_entry = entry.copy()
        
        try:
            if strategy == "typo_only":
                # Only fix typos
                cleaned_text = self.fix_typos_with_qwen(output_text)
            elif strategy == "convert_only":
                # Only convert sentence types
                cleaned_text = self.convert_to_statement_with_qwen(output_text)
            elif strategy == "filter":
                # Judge quality and filter
                if not self.judge_quality_with_qwen(output_text):
                    return None
                cleaned_text = output_text
            elif strategy == "comprehensive":
                # Comprehensive cleaning
                cleaned_text = self.comprehensive_clean_with_qwen(output_text)
            else:
                raise ValueError(f"Unknown cleaning strategy: {strategy}")
            
            if cleaned_text is None:
                return None
            
            cleaned_entry['cleaned_output'] = cleaned_text
            return cleaned_entry
            
        except Exception as e:
            logger.error(f"Error cleaning entry: {e}")
            return entry  # Return original entry on error
    
    def clean_dataset_batch(self, entries: List[Dict[str, Any]], strategy: str = "comprehensive") -> List[Dict[str, Any]]:
        """Clean data entries in batches using batch inference"""
        cleaned_entries = []
        
        # Extract texts and prepare for batch processing
        valid_entries = []
        texts_to_clean = []
        
        for entry in entries:
            if 'output' not in entry:
                cleaned_entries.append(entry)
                continue
                
            output_text = entry['output']
            if not isinstance(output_text, str) or not output_text.strip():
                cleaned_entries.append(entry)
                continue
                
            valid_entries.append(entry)
            texts_to_clean.append(output_text)
        
        if not texts_to_clean:
            return cleaned_entries
        
        # Prepare prompts based on strategy
        prompts = []
        for text in texts_to_clean:
            if strategy == "typo_only":
                prompt = self.cleaning_prompts["typo_fix"].format(text=text)
            elif strategy == "convert_only":
                prompt = self.cleaning_prompts["convert_to_statement"].format(text=text)
            elif strategy == "filter":
                prompt = self.cleaning_prompts["judge_quality"].format(text=text)
            elif strategy == "comprehensive":
                prompt = self.cleaning_prompts["comprehensive_clean"].format(text=text)
            else:
                raise ValueError(f"Unknown cleaning strategy: {strategy}")
            prompts.append(prompt)
        
        # Process in sub-batches to manage memory
        for i in range(0, len(prompts), self.inference_batch_size):
            sub_prompts = prompts[i:i + self.inference_batch_size]
            sub_entries = valid_entries[i:i + self.inference_batch_size]
            
            # Batch inference
            results = self.call_qwen_batch(sub_prompts)
            
            # Process results
            for entry, result, original_text in zip(sub_entries, results, texts_to_clean[i:i + self.inference_batch_size]):
                try:
                    if strategy == "filter":
                        # For filter strategy, check if we should keep the entry
                        if result and "KEEP" in result.upper():
                            cleaned_entry = entry.copy()
                            cleaned_entry['cleaned_output'] = original_text
                            cleaned_entries.append(cleaned_entry)
                        # If FILTER or no result, skip this entry
                    else:
                        # For other strategies, use the cleaned result
                        if result and result.upper() != "DELETE":
                            cleaned_entry = entry.copy()
                            cleaned_entry['cleaned_output'] = result
                            cleaned_entries.append(cleaned_entry)
                        # If DELETE or no result, skip this entry
                        
                except Exception as e:
                    logger.error(f"Error processing entry result: {e}")
                    # On error, keep original entry
                    cleaned_entries.append(entry)
        
        return cleaned_entries
    
    def load_and_clean_jsonl(self, input_file: str, output_file: str, strategy: str = "comprehensive",
                           batch_size: int = 100) -> None:
        """Load and clean entire jsonl file"""
        logger.info(f"Starting Qwen cleaning for file: {input_file}")
        logger.info(f"Using strategy: {strategy}")
        logger.info(f"Batch size: {batch_size}")
        
        # Read all data
        entries = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing error at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(entries)} entries")
        
        # Process in batches with global progress bar
        all_cleaned_entries = []
        with tqdm(total=len(entries), desc="Cleaning entries") as pbar:
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(entries) + batch_size - 1)//batch_size}")
                
                try:
                    cleaned_batch = self.clean_dataset_batch(batch, strategy)
                    all_cleaned_entries.extend(cleaned_batch)
                    
                    # Update progress bar
                    pbar.update(len(batch))
                    
                    # Clear GPU cache periodically to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # On batch error, try to process entries individually
                    logger.info("Falling back to individual processing for this batch")
                    for entry in batch:
                        try:
                            cleaned_entry = self.clean_single_entry(entry, strategy)
                            if cleaned_entry is not None:
                                all_cleaned_entries.append(cleaned_entry)
                        except Exception as individual_e:
                            logger.error(f"Error processing individual entry: {individual_e}")
                            # Keep original entry on error
                            all_cleaned_entries.append(entry)
                    
                    # Update progress bar for fallback processing
                    pbar.update(len(batch))
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_cleaned_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Cleaning completed!")
        logger.info(f"Original entries: {len(entries)}")
        logger.info(f"Cleaned entries: {len(all_cleaned_entries)}")
        logger.info(f"Cleaned file saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Clean dialogue dataset using GPT or Qwen')
    parser.add_argument('--input', '-i', required=True, help='Input jsonl file path')
    parser.add_argument('--output', '-o', required=True, help='Output jsonl file path')
    parser.add_argument('--cleaner-type', choices=['gpt', 'qwen'], default='gpt', 
                       help='Type of cleaner to use: gpt or qwen')
    parser.add_argument('--strategy', '-s', 
                       choices=['typo_only', 'convert_only', 'filter', 'comprehensive'],
                       default='comprehensive', 
                       help='Cleaning strategy')
    
    # GPT-specific arguments
    parser.add_argument('--api-key', help='OpenAI API key (for GPT cleaner)')
    parser.add_argument('--base-url', help='API base URL (for GPT cleaner)')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='Model to use (for GPT cleaner)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing (for GPT cleaner)')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of worker threads for parallel processing (for GPT cleaner)')
    
    # Qwen-specific arguments
    parser.add_argument('--qwen-model-path', default='Qwen/Qwen3-8B', 
                       help='Path to Qwen model or HuggingFace model name (for Qwen cleaner)')
    parser.add_argument('--device', default='auto', help='Device to use for Qwen inference (auto, cuda, cpu)')
    parser.add_argument('--inference-batch-size', type=int, default=8, 
                       help='Batch size for model inference (adjust based on GPU memory)')
    
    # Common arguments
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    
    args = parser.parse_args()
    
    try:
        if args.cleaner_type == 'gpt':
            logger.info("Using GPT cleaner")
            cleaner = GPTDatasetCleaner(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model
            )
            
            cleaner.load_and_clean_jsonl(
                args.input, 
                args.output, 
                args.strategy,
                args.parallel,
                args.max_workers,
                args.batch_size
            )
            
        elif args.cleaner_type == 'qwen':
            logger.info("Using Qwen cleaner")
            cleaner = QwenDatasetCleaner(
                model_path=args.qwen_model_path,
                device=args.device,
                inference_batch_size=args.inference_batch_size
            )
            
            cleaner.load_and_clean_jsonl(
                args.input, 
                args.output, 
                args.strategy,
                args.batch_size
            )
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
