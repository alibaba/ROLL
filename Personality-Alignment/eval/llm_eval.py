"""
LLM Evaluator for evaluating dialogue model outputs.
Based on the GPT and Qwen cleaner structure from clean_dataset.py.
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


@dataclass
class EvaluationResult:
    """Evaluation result data structure"""
    topic_alignment: int
    persona_consistency: int
    preference_consistency: int
    history_consistency: int
    explanations: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_alignment": self.topic_alignment,
            "persona_consistency": self.persona_consistency,
            "preference_consistency": self.preference_consistency,
            "history_consistency": self.history_consistency,
            "explanations": self.explanations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        return cls(
            topic_alignment=data.get("topic_alignment", 0),
            persona_consistency=data.get("persona_consistency", 0),
            preference_consistency=data.get("preference_consistency", 0),
            history_consistency=data.get("history_consistency", 0),
            explanations=data.get("explanations", {})
        )


EVAL_PROMPT = """
You are an impartial dialogue‑quality evaluator.

### Inputs
• Persona profile  
{PROFILE}

• Conversation history (most recent turn last)  
{CONVERSATION_HISTORY}

• Ground‑truth reply (the actual answer written by a human in the persona)  
{GROUND_TRUTH}

• Model‑generated reply to be evaluated  
{MODEL_OUTPUT}

### Evaluation Criteria
Score each criterion strictly within its numeric range. Use the full scale whenever warranted.

1. **Topic Alignment with Ground‑Truth** (1–10)  
   *10 = conveys essentially the same topic, intent, and key points as the ground‑truth;  
   1 = off‑topic or contradicts it.*

2. **Persona Consistency** (1–3)  
   *3 = strongly reflects the self‑description (tone, values, wording);  
   1 = no discernible match or outright contradiction.*

3. **Preference Consistency** (1–3)  
   *3 = clearly respects / incorporates the stated preferences;  
   1 = ignores or conflicts with them.*

4. **Conversation‑History Consistency** (1–3)  
   *3 = seamlessly follows from, references, and does not contradict prior turns;  
   1 = breaks continuity or introduces contradictions.*

### Instructions
1. Read all inputs carefully before scoring.  
2. Provide a **brief justification** (1–3 sentences) for each score.  
3. Return your verdict **only** in the JSON schema below.

### Output JSON schema
{
  "topic_alignment": <integer 1‑10>,
  "persona_consistency": <integer 1‑3>,
  "preference_consistency": <integer 1‑3>,
  "history_consistency": <integer 1‑3>,
  "explanations": {
    "topic_alignment": "<one‑sentence rationale>",
    "persona_consistency": "<one‑sentence rationale>",
    "preference_consistency": "<one‑sentence rationale>",
    "history_consistency": "<one‑sentence rationale>"
  }
}
"""


class GPTEvaluator:
    """Class for evaluating dialogue outputs using GPT"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize GPT evaluator
        
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
    
    def evaluate_single_output(self, profile: str, conversation_history: str, 
                             ground_truth: str, model_output: str) -> Optional[EvaluationResult]:
        """Evaluate a single model output"""
        prompt = EVAL_PROMPT.format(
            PROFILE=profile,
            CONVERSATION_HISTORY=conversation_history,
            GROUND_TRUTH=ground_truth,
            MODEL_OUTPUT=model_output
        )
        
        response = self.call_gpt(prompt)
        if not response:
            return None
        
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result_dict = json.loads(json_str)
                return EvaluationResult.from_dict(result_dict)
            else:
                logger.error(f"No valid JSON found in response: {response}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}, response: {response}")
            return None
    
    def evaluate_batch(self, evaluation_data: List[Dict[str, Any]]) -> List[Optional[EvaluationResult]]:
        """Evaluate a batch of model outputs"""
        results = []
        
        for data in evaluation_data:
            result = self.evaluate_single_output(
                profile=data.get('profile', ''),
                conversation_history=data.get('conversation_history', ''),
                ground_truth=data.get('ground_truth', ''),
                model_output=data.get('model_output', '')
            )
            results.append(result)
            
            # Add small delay to avoid API limits
            time.sleep(0.1)
        
        return results
    
    def evaluate_parallel(self, evaluation_data: List[Dict[str, Any]], 
                         max_workers: int = 5) -> List[Optional[EvaluationResult]]:
        """Evaluate model outputs in parallel"""
        results = [None] * len(evaluation_data)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self.evaluate_single_output,
                    data.get('profile', ''),
                    data.get('conversation_history', ''),
                    data.get('ground_truth', ''),
                    data.get('model_output', '')
                ): i for i, data in enumerate(evaluation_data)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Parallel evaluation error at index {index}: {e}")
        
        return results


class QwenEvaluator:
    """Class for evaluating dialogue outputs using Qwen"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto", inference_batch_size: int = 8):
        """
        Initialize Qwen evaluator
        
        Args:
            model_path: Path to Qwen model or HuggingFace model name
            device: Device to use for inference ('auto', 'cuda', 'cpu')
            inference_batch_size: Batch size for model inference
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
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
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
    
    def evaluate_single_output(self, profile: str, conversation_history: str, 
                             ground_truth: str, model_output: str) -> Optional[EvaluationResult]:
        """Evaluate a single model output"""
        prompt = EVAL_PROMPT.format(
            PROFILE=profile,
            CONVERSATION_HISTORY=conversation_history,
            GROUND_TRUTH=ground_truth,
            MODEL_OUTPUT=model_output
        )
        
        response = self.call_qwen(prompt)
        if not response:
            return None
        
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result_dict = json.loads(json_str)
                return EvaluationResult.from_dict(result_dict)
            else:
                logger.error(f"No valid JSON found in response: {response}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}, response: {response}")
            return None
    
    def evaluate_batch(self, evaluation_data: List[Dict[str, Any]]) -> List[Optional[EvaluationResult]]:
        """Evaluate model outputs in batches using batch inference"""
        results = []
        
        # Prepare prompts
        prompts = []
        for data in evaluation_data:
            prompt = EVAL_PROMPT.format(
                PROFILE=data.get('profile', ''),
                CONVERSATION_HISTORY=data.get('conversation_history', ''),
                GROUND_TRUTH=data.get('ground_truth', ''),
                MODEL_OUTPUT=data.get('model_output', '')
            )
            prompts.append(prompt)
        
        # Process in sub-batches to manage memory
        for i in range(0, len(prompts), self.inference_batch_size):
            sub_prompts = prompts[i:i + self.inference_batch_size]
            
            # Batch inference
            responses = self.call_qwen_batch(sub_prompts)
            
            # Parse results
            for response in responses:
                if not response:
                    results.append(None)
                    continue
                
                try:
                    # Try to extract JSON from response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result_dict = json.loads(json_str)
                        results.append(EvaluationResult.from_dict(result_dict))
                    else:
                        logger.error(f"No valid JSON found in response: {response}")
                        results.append(None)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}, response: {response}")
                    results.append(None)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results


class LLMEvaluator:
    """Main evaluator class that can use either GPT or Qwen"""
    
    def __init__(self, evaluator_type: str = "gpt", **kwargs):
        """
        Initialize LLM evaluator
        
        Args:
            evaluator_type: Type of evaluator to use ('gpt' or 'qwen')
            **kwargs: Additional arguments passed to the specific evaluator
        """
        self.evaluator_type = evaluator_type
        
        if evaluator_type == "gpt":
            self.evaluator = GPTEvaluator(**kwargs)
        elif evaluator_type == "qwen":
            self.evaluator = QwenEvaluator(**kwargs)
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    def evaluate_dataset(self, input_file: str, output_file: str, parallel: bool = False, 
                        max_workers: int = 5, batch_size: int = 100) -> Dict[str, float]:
        """
        Evaluate an entire dataset and save results
        
        Args:
            input_file: Input jsonl file with evaluation data
            output_file: Output jsonl file for results
            parallel: Whether to use parallel processing (GPT only)
            max_workers: Number of workers for parallel processing
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with evaluation statistics
        """
        logger.info(f"Starting evaluation with {self.evaluator_type} evaluator")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        
        # Read evaluation data
        evaluation_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    evaluation_data.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing error at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(evaluation_data)} entries for evaluation")
        
        # Process in batches
        all_results = []
        successful_evaluations = 0
        
        with tqdm(total=len(evaluation_data), desc="Evaluating entries") as pbar:
            for i in range(0, len(evaluation_data), batch_size):
                batch = evaluation_data[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(evaluation_data) + batch_size - 1)//batch_size}")
                
                try:
                    if self.evaluator_type == "gpt" and parallel:
                        batch_results = self.evaluator.evaluate_parallel(batch, max_workers)
                    else:
                        batch_results = self.evaluator.evaluate_batch(batch)
                    
                    # Combine original data with evaluation results
                    for j, (original_entry, result) in enumerate(zip(batch, batch_results)):
                        output_entry = original_entry.copy()
                        if result:
                            output_entry['evaluation_result'] = result.to_dict()
                            successful_evaluations += 1
                        else:
                            output_entry['evaluation_result'] = None
                        all_results.append(output_entry)
                    
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Add entries without evaluation results
                    for original_entry in batch:
                        output_entry = original_entry.copy()
                        output_entry['evaluation_result'] = None
                        all_results.append(output_entry)
                    pbar.update(len(batch))
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_results:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        # Calculate statistics
        stats = self._calculate_statistics(all_results)
        
        logger.info(f"Evaluation completed!")
        logger.info(f"Total entries: {len(evaluation_data)}")
        logger.info(f"Successful evaluations: {successful_evaluations}")
        logger.info(f"Success rate: {successful_evaluations/len(evaluation_data)*100:.1f}%")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Evaluation statistics: {stats}")
        
        return stats
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation statistics"""
        valid_results = [r for r in results if r.get('evaluation_result') is not None]
        
        if not valid_results:
            return {}
        
        # Extract scores
        topic_scores = [r['evaluation_result']['topic_alignment'] for r in valid_results]
        persona_scores = [r['evaluation_result']['persona_consistency'] for r in valid_results]
        preference_scores = [r['evaluation_result']['preference_consistency'] for r in valid_results]
        history_scores = [r['evaluation_result']['history_consistency'] for r in valid_results]
        
        # Calculate statistics
        stats = {
            'num_evaluated': len(valid_results),
            'topic_alignment_mean': np.mean(topic_scores),
            'topic_alignment_std': np.std(topic_scores),
            'persona_consistency_mean': np.mean(persona_scores),
            'persona_consistency_std': np.std(persona_scores),
            'preference_consistency_mean': np.mean(preference_scores),
            'preference_consistency_std': np.std(preference_scores),
            'history_consistency_mean': np.mean(history_scores),
            'history_consistency_std': np.std(history_scores)
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate dialogue model outputs using LLM evaluators')
    parser.add_argument('--input', '-i', required=True, help='Input jsonl file with evaluation data')
    parser.add_argument('--output', '-o', required=True, help='Output jsonl file for results')
    parser.add_argument('--evaluator-type', choices=['gpt', 'qwen'], default='gpt',
                       help='Type of evaluator to use: gpt or qwen')
    
    # GPT-specific arguments
    parser.add_argument('--api-key', help='OpenAI API key (for GPT evaluator)')
    parser.add_argument('--base-url', help='API base URL (for GPT evaluator)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use (for GPT evaluator)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing (for GPT evaluator)')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of worker threads')
    
    # Qwen-specific arguments
    parser.add_argument('--qwen-model-path', default='Qwen/Qwen2.5-7B-Instruct',
                       help='Path to Qwen model or HuggingFace model name')
    parser.add_argument('--device', default='auto', help='Device to use for Qwen inference')
    parser.add_argument('--inference-batch-size', type=int, default=8,
                       help='Batch size for model inference')
    
    # Common arguments
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    
    args = parser.parse_args()
    
    try:
        if args.evaluator_type == 'gpt':
            evaluator = LLMEvaluator(
                evaluator_type='gpt',
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model
            )
            
            stats = evaluator.evaluate_dataset(
                args.input,
                args.output,
                args.parallel,
                args.max_workers,
                args.batch_size
            )
            
        elif args.evaluator_type == 'qwen':
            evaluator = LLMEvaluator(
                evaluator_type='qwen',
                model_path=args.qwen_model_path,
                device=args.device,
                inference_batch_size=args.inference_batch_size
            )
            
            stats = evaluator.evaluate_dataset(
                args.input,
                args.output,
                False,  # Qwen doesn't support parallel processing
                args.max_workers,
                args.batch_size
            )
        
        # Print final statistics
        if stats:
            print("\n=== Evaluation Statistics ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

