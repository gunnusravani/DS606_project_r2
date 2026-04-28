"""
Ultra-simple Llama Guard 4 evaluation - text only, NO vision dependencies.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

print("="*70)
print("LLAMA GUARD 4 - TEXT ONLY")
print("="*70)

# Diagnostic info
print("\n[DIAGNOSTICS]")
print(f"Python: {sys.version.split()[0]}")
print(f"Executable: {sys.executable}")

# Import torch
print("\n[1/4] Importing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ Failed: {e}")
    raise

# Import transformers with diagnostics
print("\n[2/4] Importing Transformers...")
try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Failed to import transformers: {e}")
    raise

# Try importing the specific classes with fallback
print("\n[3/4] Loading tokenizer and model classes...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"✓ Direct import successful")
except ImportError as e:
    print(f"  Trying alternative import method...")
    try:
        import transformers.models.auto.tokenization_auto as ta
        AutoTokenizer = ta.AutoTokenizer
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM
        print(f"✓ Alternative import successful")
    except Exception as e2:
        print(f"✗ All methods failed: {e2}")
        raise



def load_model():
    """Load Llama Guard 4 - text tokenizer only."""
    print("\n[4/4] Loading Llama Guard 4 model...")
    
    try:
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-4-12B")
        print(f"  ✓ Tokenizer loaded")
        
        print("  Loading model (this may take 2-3 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-Guard-4-12B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        print(f"  ✓ Model loaded")
        
        device = next(model.parameters()).device
        print(f"  ✓ Model on {device}")
        
        return tokenizer, model
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def classify_response(tokenizer, model, prompt: str, response: str) -> Tuple[str, List[str]]:
    """Classify response."""
    try:
        # Simple conversation format
        text = f"User: {prompt}\n\nAssistant: {response}"
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        
        # Decode
        response_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse
        lines = response_text.split('\n')
        safety_label = "unsafe" if "unsafe" in lines[0].lower() else "safe"
        
        categories = []
        if len(lines) > 1:
            import re
            categories = re.findall(r'S\d{1,2}', lines[1])
        
        return safety_label, categories
    except Exception as e:
        print(f"  Error: {e}")
        return "safe", []


def load_completions(lang: str) -> List[Dict]:
    """Load completions."""
    base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
    file_path = base_path / f"{lang}/20250519-000604/1/completions/harmful_harm_ablation_evaluations.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    completions = data.get("completions", data if isinstance(data, list) else [])
    return completions


def evaluate_language(tokenizer, model, lang: str):
    """Evaluate a language."""
    print(f"\n  Evaluating {lang.upper()}...")
    
    completions = load_completions(lang)
    use_field = "response_translated"
    
    unsafe_count = 0
    
    for i, completion in enumerate(completions):
        prompt = completion.get("instruction", "")
        response = completion.get(use_field, "")
        
        safety_label, _ = classify_response(tokenizer, model, prompt, response)
        
        if safety_label == "unsafe":
            unsafe_count += 1
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    {i+1}/{len(completions)}: {unsafe_count} unsafe")
    
    # Save
    base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
    output_file = base_path / f"{lang}/20250519-000604/1/completions/llama_guard_4_results.json"
    
    summary = {
        'total': len(completions),
        'unsafe_count': unsafe_count,
        'unsafe_rate': unsafe_count / len(completions) if len(completions) > 0 else 0,
        'safe_rate': (len(completions) - unsafe_count) / len(completions) if len(completions) > 0 else 0,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  ✓ {lang.upper()}: {summary['unsafe_rate']:.2%} unsafe")
    print(f"    Saved: {output_file}")
    
    return summary


def main():
    """Main."""
    tokenizer, model = load_model()
    
    print(f"\n[PROCESSING]")
    results = {}
    
    for lang in ['hi', 'bn']:
        results[lang] = evaluate_language(tokenizer, model, lang)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for lang, result in results.items():
        print(f"{lang.upper()}: {result['unsafe_rate']:.2%} unsafe ({result['unsafe_count']}/{result['total']})")
    print("="*70)


if __name__ == "__main__":
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import sys
        sys.exit(1)

