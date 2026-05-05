"""
Extended script to identify refusal tokens for multiple models and languages.

Supports:
  - Llama-3.1-8B-Instruct (hi, bn)
  - Qwen2.5-7B-Instruct (hi, bn)
  - Gemma-2-9B-it (hi, bn)

Tokens are identified by generating responses to harmful prompts and extracting
the first tokens produced, which typically indicate refusal responses.
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

<<<<<<< Updated upstream
# # Load HF_TOKEN from environment
# hf_token = os.getenv('HF_TOKEN')
# print(hf_token)
# if hf_token:
#     login(token=hf_token)
#     print(f"✅ Authenticated with HuggingFace using HF_TOKEN from environment")
# else:
#     print("⚠️  HF_TOKEN not found in environment.")
=======
# Load HF_TOKEN from environment
hf_token = os.getenv('HF_TOKEN')
offline_mode = os.getenv('HF_HUB_OFFLINE') == '1' or os.getenv('TRANSFORMERS_OFFLINE') == '1'
if hf_token and not offline_mode:
    login(token=hf_token)
    print(f"✅ Authenticated with HuggingFace using HF_TOKEN from environment")
elif offline_mode:
    print("⚠️  Offline mode enabled; skipping Hugging Face login and using cached files only.")
else:
    print("⚠️  HF_TOKEN not found in environment.")
>>>>>>> Stashed changes

# Model configurations
MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

LANGUAGES = ("hi", "bn")

# Chat templates per model
CHAT_TEMPLATES = {
    "llama3.1-8b": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "qwen2.5-7b": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    "gemma2-9b": "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
}


def format_instruction_chat(model_key: str, instruction: str) -> str:
    """Format instruction in model-specific chat template."""
    template = CHAT_TEMPLATES.get(model_key, "")
    return template.format(instruction=instruction)


def tokenize_instructions(tokenizer, model_key: str, instructions: List[str]) -> Dict:
    """Tokenize instructions using appropriate chat template."""
    prompts = [format_instruction_chat(model_key, instruction) for instruction in instructions]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return result


def identify_refusal_tokens_for_model(
    model_key: str,
    model_path: str,
    harmful_prompts_file: str,
    lang_code: str,
    device: str = "cuda",
    sample_size: int = 100,
) -> List[int]:
    """
    Identify refusal tokens by generating responses to harmful prompts.
    
    Args:
        model_key: Key for chat template (e.g., "qwen2.5-7b")
        model_path: HuggingFace model path
        harmful_prompts_file: Path to JSON file with harmful prompts
        lang_code: Language code (e.g., "hi", "bn")
        device: Device to use ("cuda" or "cpu")
        sample_size: Number of prompts to sample
    
    Returns:
        List of token IDs that represent refusals
    """
    print(f"\n{'='*70}")
    print(f"Identifying refusal tokens for {model_key} / {lang_code.upper()}")
    print(f"Model: {model_path}")
    print(f"{'='*70}")
    
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    print(f"Loading harmful prompts from {harmful_prompts_file}...")
    with open(harmful_prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract instructions
    harmful_instructions = [d['instruction'] for d in data[:sample_size]]
    print(f"✅ Loaded {len(harmful_instructions)} harmful prompts")
    
    # Tokenize and generate
    tokenized = tokenize_instructions(tokenizer, model_key, harmful_instructions)
    
    print(f"Generating responses (max_new_tokens=20)...")
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
    
    print(f"\n✅ Generated responses - First tokens:")
    print(f"Unique first token IDs: {unique_first_tokens.tolist()}")
    
    # Decode tokens to show what they are
    for token_id in unique_first_tokens:
        token_str = tokenizer.decode(token_id)
        print(f"  Token ID {token_id.item()}: '{token_str}'")
    
    # Show some examples
    print(f"\n📝 Sample responses:")
    for i in range(min(3, len(generation_toks))):
        response_tokens = generation_toks[i, input_len:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        print(f"  Prompt {i+1}: {harmful_instructions[i][:80]}...")
        print(f"  Response: {response[:150]}...\n")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return unique_first_tokens.tolist()


def main():
    print(f"\n{'='*70}")
    print("MULTI-MODEL REFUSAL TOKEN IDENTIFICATION")
    print(f"{'='*70}")
    
    BASE_DIR = Path(__file__).parent.parent
    config_dir = BASE_DIR / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all tokens organized by model and language
    all_tokens: Dict[str, Dict[str, List[int]]] = {}
    
    # Iterate over all model-language pairs
    for model_key, model_path in MODELS.items():
        all_tokens[model_key] = {}
        
        for lang in LANGUAGES:
            # Determine the correct dataset file
            dataset_file = BASE_DIR / f"dataset/splits_multi/harmful_train_translated_{lang}.json"
            
            if not dataset_file.exists():
                print(f"⚠️  Dataset file not found: {dataset_file}")
                print(f"   Attempting alternative path...")
                dataset_file = BASE_DIR / f"dataset/raw/harmful_train_translated_{lang}.json"
                if not dataset_file.exists():
                    print(f"❌ Could not find dataset for {lang}, skipping")
                    continue
            
            # Identify tokens
            tokens = identify_refusal_tokens_for_model(
                model_key=model_key,
                model_path=model_path,
                harmful_prompts_file=str(dataset_file),
                lang_code=lang,
                sample_size=100,
            )
            
            all_tokens[model_key][lang] = tokens
            print(f"✅ Identified {len(tokens)} refusal tokens for {model_key}/{lang}")
    
    # Save all tokens to file
    tokens_file = config_dir / "refusal_tokens_multimodel.json"
    all_tokens_data = {
        "tokens": all_tokens,
        "timestamp": str(Path.cwd()),
        "note": "Refusal tokens for multiple models and languages",
        "models": list(MODELS.keys()),
        "languages": list(LANGUAGES),
    }
    
    with open(tokens_file, 'w', encoding='utf-8') as f:
        json.dump(all_tokens_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ REFUSAL TOKENS SAVED TO: {tokens_file}")
    print(f"{'='*70}\n")
    
    print("📊 Summary:")
    for model_key, langs in all_tokens.items():
        for lang, tokens in langs.items():
            print(f"  {model_key:15} / {lang:3} → {len(tokens)} tokens: {tokens}")
    
    print("\n✅ Next steps:")
    print(f"   1. Review config/refusal_tokens_multimodel.json")
    print(f"   2. Update model wrappers if needed with new tokens")
    print(f"   3. Run generation script:")
    print(f"      python scripts/generate_with_refusal_vectors_extended.py --operation add,subtract --projection same,cross")


if __name__ == "__main__":
    main()
