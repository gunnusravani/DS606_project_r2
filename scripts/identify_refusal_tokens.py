"""
Script to identify refusal tokens for Hindi and Bengali languages.
Refusal tokens are the first tokens that appear when the model refuses a harmful request.
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

def format_instruction_qwen_chat(instruction: str, system: str = None) -> str:
    """Format instruction in Qwen chat template format."""
    QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    return QWEN_CHAT_TEMPLATE.format(instruction=instruction)

def tokenize_instructions(tokenizer, instructions: List[str]) -> Dict:
    """Tokenize instructions using Qwen chat template."""
    prompts = [
        format_instruction_qwen_chat(instruction=instruction)
        for instruction in instructions
    ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return result

def identify_refusal_tokens(model_path: str, harmful_prompts_file: str, lang_code: str, device: str = "cuda"):
    """
    Identify refusal tokens by generating responses to harmful prompts.
    
    Args:
        model_path: Path to the model (e.g., "Qwen/Qwen2.5-7B-Instruct")
        harmful_prompts_file: Path to JSON file with harmful prompts in the target language
        lang_code: Language code (e.g., "hi", "bn")
        device: Device to use ("cuda" or "cpu")
    
    Returns:
        List of token IDs that represent refusals in the target language
    """
    print(f"Loading model {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading harmful prompts from {harmful_prompts_file}...")
    with open(harmful_prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract instructions
    harmful_instructions = [d['instruction'] for d in data[:5]]  # Sample first 5
    print(f"Loaded {len(harmful_instructions)} harmful prompts")
    
    # Tokenize and generate
    tokenized = tokenize_instructions(tokenizer, harmful_instructions)
    
    print("Generating responses...")
    with torch.no_grad():
        generation_toks = model.generate(
            input_ids=tokenized.input_ids.to(model.device),
            attention_mask=tokenized.attention_mask.to(model.device),
            max_new_tokens=20,
            do_sample=False,
        )
    
    # Extract first tokens (after input)
    input_len = tokenized.input_ids.shape[-1]
    first_tokens = generation_toks[:, input_len:input_len+1]
    unique_first_tokens = torch.unique(first_tokens)
    
    print(f"\nGenerated responses - First tokens:")
    print(f"Unique first token IDs: {unique_first_tokens.tolist()}")
    
    # Decode tokens to show what they are
    for token_id in unique_first_tokens:
        token_str = tokenizer.decode(token_id)
        print(f"  Token ID {token_id.item()}: '{token_str}'")
    
    # Show some examples
    print(f"\nSample responses:")
    for i in range(min(3, len(generation_toks))):
        response_tokens = generation_toks[i, input_len:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        print(f"  Prompt {i+1}: {harmful_instructions[i][:80]}...")
        print(f"  Response: {response[:150]}...\n")
    
    return unique_first_tokens.tolist()

if __name__ == "__main__":
    # Example usage for Hindi
    print("=" * 80)
    print("Identifying refusal tokens for Hindi (hi)")
    print("=" * 80)
    hi_tokens = identify_refusal_tokens(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        harmful_prompts_file="/Users/sravani/Documents/VSCode_projects/Multilingual-Refusal/dataset/splits_multi/harmful_train_translated_hi.json",
        lang_code="hi"
    )
    print(f"\nRecommended Hindi refusal tokens: {hi_tokens}")
    
    print("\n" + "=" * 80)
    print("Identifying refusal tokens for Bengali (bn)")
    print("=" * 80)
    bn_tokens = identify_refusal_tokens(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        harmful_prompts_file="/Users/sravani/Documents/VSCode_projects/Multilingual-Refusal/dataset/splits_multi/harmful_train_translated_bn.json",
        lang_code="bn"
    )
    print(f"\nRecommended Bengali refusal tokens: {bn_tokens}")
