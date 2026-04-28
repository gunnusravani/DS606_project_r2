"""
Simple Llama Guard 4 evaluation script.
No complexity, just load model and evaluate responses.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple

# Test PyTorch version
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load model with minimal dependencies
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("✓ Transformers imported successfully")
except Exception as e:
    print(f"✗ Failed to import transformers: {e}")
    raise


def load_model():
    """Load Llama Guard 4 model."""
    print("\nLoading Llama Guard 4 model...")
    processor = AutoProcessor.from_pretrained("meta-llama/Llama-Guard-4-12B")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-Guard-4-12B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("✓ Model loaded successfully")
    return processor, model


def classify_response(processor, model, prompt: str, response: str) -> Tuple[str, List[str]]:
    """Classify a single response."""
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


def load_completions(lang: str) -> List[Dict]:
    """Load completions from JSON."""
    base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
    file_path = base_path / f"{lang}/20250519-000604/1/completions/harmful_harm_ablation_evaluations.json"
    
    print(f"\nLoading {lang} completions from: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    completions = data.get("completions", data if isinstance(data, list) else [])
    print(f"✓ Loaded {len(completions)} completions")
    
    return completions


def evaluate_language(processor, model, lang: str):
    """Evaluate all completions for a language."""
    print(f"\n{'='*70}")
    print(f"Evaluating {lang.upper()}")
    print(f"{'='*70}")
    
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
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(completions)}")
    
    # Save results
    base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
    output_file = base_path / f"{lang}/20250519-000604/1/completions/llama_guard_4_results.json"
    
    summary = {
        'total': len(completions),
        'unsafe_count': unsafe_count,
        'unsafe_rate': unsafe_count / len(completions),
        'safe_rate': (len(completions) - unsafe_count) / len(completions),
        'details': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ {lang.upper()} Results:")
    print(f"  Unsafe Rate: {summary['unsafe_rate']:.2%}")
    print(f"  Safe Rate: {summary['safe_rate']:.2%}")
    print(f"  Saved to: {output_file}")
    
    return summary


def main():
    """Main execution."""
    print("="*70)
    print("SIMPLE LLAMA GUARD 4 EVALUATION")
    print("="*70)
    
    # Load model once
    print("\nStep 1: Loading model...")
    processor, model = load_model()
    
    # Evaluate both languages
    print("\nStep 2: Evaluating languages...")
    results = {}
    
    for lang in ['hi', 'bn']:
        try:
            results[lang] = evaluate_language(processor, model, lang)
        except Exception as e:
            print(f"✗ Error evaluating {lang}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for lang, result in results.items():
        print(f"\n{lang.upper()}:")
        print(f"  Unsafe Rate: {result['unsafe_rate']:.2%}")
        print(f"  Safe Rate: {result['safe_rate']:.2%}")


if __name__ == "__main__":
    main()
