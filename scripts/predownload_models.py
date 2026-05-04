"""
Pre-download models to local cache on a machine with internet access.

Run this on a login node BEFORE submitting SLURM jobs.

Usage:
    python scripts/predownload_models.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

def predownload_model(model_name: str, model_path: str) -> None:
    """Download and cache a model locally."""
    print(f"\n{'='*70}")
    print(f"Downloading: {model_name} ({model_path})")
    print(f"{'='*70}")
    
    try:
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  ✓ Tokenizer cached")
        
        print(f"  Loading model (dtype=bfloat16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"  ✓ Model cached")
        
        # Clean up
        del model
        del tokenizer
        print(f"  ✓ {model_name} ready for compute nodes\n")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}\n")
        raise

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"MODEL PRE-DOWNLOAD SCRIPT")
    print(f"{'='*70}")
    print(f"This downloads {len(MODELS)} models to local HF cache")
    print(f"Run this on a login node with internet access")
    print(f"Models will be cached for compute nodes to use")
    print(f"{'='*70}\n")
    
    for model_name, model_path in MODELS.items():
        predownload_model(model_name, model_path)
    
    print(f"{'='*70}")
    print(f"✓ ALL MODELS DOWNLOADED AND CACHED")
    print(f"{'='*70}")
    print(f"Your HuggingFace cache is at: ~/.cache/huggingface/hub/")
    print(f"This cache is now ready for compute nodes to use.\n")
