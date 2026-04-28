"""
Simple Llama Guard 4 evaluation script.
No complexity, just load model and evaluate responses.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

print("="*70)
print("SIMPLE LLAMA GUARD 4 EVALUATION")
print("="*70)

# Import torch first
print("\n[1/5] Importing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")
    raise

# Import transformers
print("\n[2/5] Importing Transformers...")
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("✓ Transformers imported successfully")
except Exception as e:
    print(f"✗ Failed to import transformers: {e}")
    print("\nTrying alternative import strategy...")
    try:
        import transformers
        print(f"  Transformers version: {transformers.__version__}")
        from transformers import AutoProcessor, AutoModelForCausalLM
        print("✓ Alternative import successful")
    except Exception as e2:
        print(f"✗ Alternative import also failed: {e2}")
        raise


def load_model():
    """Load Llama Guard 4 model."""
    print("\n[3/5] Loading Llama Guard 4 model (this may take 1-2 minutes)...")
    
    try:
        print("  Loading processor...")
        processor = AutoProcessor.from_pretrained("meta-llama/Llama-Guard-4-12B")
        print("  ✓ Processor loaded")
        
        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-Guard-4-12B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("  ✓ Model loaded")
        
        # Verify device
        device = next(model.parameters()).device
        print(f"  Model on device: {device}")
        
        return processor, model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise


def classify_response(processor, model, prompt: str, response: str) -> Tuple[str, List[str]]:
    """Classify a single response."""
    try:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        response_text = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )[0]
        
        # Parse response
        lines = response_text.strip().split('\n')
        safety_label = "unsafe" if "unsafe" in lines[0].lower() else "safe"
        categories = []
        
        if len(lines) > 1:
            import re
            categories = re.findall(r'S\d{1,2}', lines[1])
        
        return safety_label, categories
    except Exception as e:
        print(f"    ✗ Error classifying: {e}")
        return "safe", []


def load_completions(lang: str) -> List[Dict]:
    """Load completions from JSON."""
    base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
    file_path = base_path / f"{lang}/20250519-000604/1/completions/harmful_harm_ablation_evaluations.json"
    
    print(f"  Loading from: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    completions = data.get("completions", data if isinstance(data, list) else [])
    print(f"  ✓ Loaded {len(completions)} completions")
    
    return completions


def evaluate_language(processor, model, lang: str):
    """Evaluate all completions for a language."""
    print(f"\n[4/5] Evaluating {lang.upper()}...")
    
    completions = load_completions(lang)
    use_field = "response_translated"
    
    results = []
    unsafe_count = 0
    
    for i, completion in enumerate(completions):
        prompt = completion.get("instruction", "")
        response = completion.get(use_field, "")
        
        safety_label, categories = classify_response(processor, model, prompt, response)
        
        results.append({
            'index': i,
            'safety_label': safety_label,
            'categories': categories,
        })
        
        if safety_label == "unsafe":
            unsafe_count += 1
        
        # Print progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Processed {i+1}/{len(completions)}: {unsafe_count} unsafe so far")
    
    # Save results
    base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
    output_file = base_path / f"{lang}/20250519-000604/1/completions/llama_guard_4_results.json"
    
    summary = {
        'total': len(completions),
        'unsafe_count': unsafe_count,
        'unsafe_rate': unsafe_count / len(completions) if len(completions) > 0 else 0,
        'safe_rate': (len(completions) - unsafe_count) / len(completions) if len(completions) > 0 else 0,
        'details': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ {lang.upper()} Results:")
    print(f"  Total: {summary['total']}")
    print(f"  Unsafe: {summary['unsafe_count']}")
    print(f"  Unsafe Rate: {summary['unsafe_rate']:.2%}")
    print(f"  Safe Rate: {summary['safe_rate']:.2%}")
    print(f"  Saved to: {output_file}")
    
    return summary


def main():
    """Main execution."""
    
    # Load model once
    processor, model = load_model()
    
    # Evaluate both languages
    print("\n[5/5] Evaluating both languages...")
    results = {}
    
    for lang in ['hi', 'bn']:
        try:
            results[lang] = evaluate_language(processor, model, lang)
        except Exception as e:
            print(f"✗ Error evaluating {lang}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for lang, result in results.items():
        print(f"\n{lang.upper()}:")
        print(f"  Unsafe Rate: {result['unsafe_rate']:.2%} ({result['unsafe_count']}/{result['total']})")
        print(f"  Safe Rate: {result['safe_rate']:.2%}")
    
    print("\n✓✓✓ EVALUATION COMPLETE ✓✓✓\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ EXECUTION FAILED: {e}")
        import sys
        sys.exit(1)

