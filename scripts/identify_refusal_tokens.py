"""
Script to identify refusal tokens for Hindi and Bengali languages.
Refusal tokens are the first tokens that appear when the model refuses a harmful request.
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load .env from current directory and parent directories
    env_path = Path('.env')
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        env_path = Path.home() / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from: {env_path}")
    else:
        print("⚠️  .env file not found, using environment variables")
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env loading")

# Loading models from cached HuggingFace directory (e.g., /janaki/common/huggingface/)
print(f"ℹ️  Loading models locally from HuggingFace cache (offline mode)")
print(f"   HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE', '0')}")
print(f"   TRANSFORMERS_OFFLINE={os.getenv('TRANSFORMERS_OFFLINE', '0')}")
print(f"   HF_HUB_CACHE={os.getenv('HF_HUB_CACHE', 'default')}")

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
    print(f"Loading model {model_path} (local_files_only=True)...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            local_files_only=True,
        )
        print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    except RuntimeError as e:
        print(f"❌ CUDA error: {e}")
        raise
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading harmful prompts from {harmful_prompts_file}...")
    with open(harmful_prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract instructions
    harmful_instructions = [d['instruction'] for d in data[:100]]  # Sample first 100
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
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
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
    import sys
    import os
    
    # Get the base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Example usage for Hindi
    print("=" * 80)
    print("Identifying refusal tokens for Hindi (hi)")
    print("=" * 80)
    hi_tokens = identify_refusal_tokens(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        harmful_prompts_file=os.path.join(BASE_DIR, "dataset/splits_multi/harmful_train_translated_hi.json"),
        lang_code="hi"
    )
    print(f"\nRecommended Hindi refusal tokens: {hi_tokens}")
    
    print("\n" + "=" * 80)
    print("Identifying refusal tokens for Bengali (bn)")
    print("=" * 80)
    bn_tokens = identify_refusal_tokens(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        harmful_prompts_file=os.path.join(BASE_DIR, "dataset/splits_multi/harmful_train_translated_bn.json"),
        lang_code="bn"
    )
    print(f"\nRecommended Bengali refusal tokens: {bn_tokens}")
    
    # Save tokens to file for later use
    tokens_data = {
        'hi': hi_tokens,
        'bn': bn_tokens,
        'timestamp': str(Path.cwd()),
        'note': 'These tokens are used in the pipeline to identify refusal positions'
    }
    
    tokens_file = os.path.join(BASE_DIR, "config/refusal_tokens_hi_bn.json")
    os.makedirs(os.path.dirname(tokens_file), exist_ok=True)
    
    with open(tokens_file, 'w', encoding='utf-8') as f:
        json.dump(tokens_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Refusal tokens saved to: {tokens_file}")
    print(f"{'='*80}")
    print(f"\n📝 Summary:")
    print(f"   Hindi tokens:    {hi_tokens}")
    print(f"   Bengali tokens:  {bn_tokens}")
    print(f"\n📌 Next step:")
    print(f"   Update model files:")
    print(f"   python scripts/update_refusal_tokens.py --hi \"{','.join(map(str, hi_tokens))}\" --bn \"{','.join(map(str, bn_tokens))}\"")
