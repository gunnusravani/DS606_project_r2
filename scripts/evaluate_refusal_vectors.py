"""
Evaluate refusal vectors across all variants (add/subtract, same/cross-lingual).

Computes ASR (Attack Success Rate) using WildGuard and Gemma-3-27B-it for:
  - All models (llama3.1-8b, qwen2.5-7b, gemma2-9b)
  - All target languages (hi, bn)
  - All operations (add, subtract)
  - All projections (same_lang, cross_hi_to_bn, cross_bn_to_hi)

Creates comparison tables and visualizations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from evaluators.gemma_3 import create_gemma_3_evaluator
from evaluators.wildguard import WildGuardEvaluator


MODELS = ["llama3.1-8b", "qwen2.5-7b", "gemma2-9b"]
TARGET_LANGUAGES = ["hi", "bn"]
OPERATIONS = ["add", "subtract"]
PROJECTIONS = ["same_lang", "cross_hi_to_bn", "cross_bn_to_hi"]

DEFAULT_COMPLETION_FILENAME = "harmful_harm_ablation_evaluations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate refusal vectors across all variants"
    )
    parser.add_argument(
        "--completions-dir",
        type=str,
        default="output/refusal_vectors",
        help="Directory containing generated completions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/refusal_vectors_evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
        help="Comma-separated models to evaluate",
    )
    parser.add_argument(
        "--target-langs",
        type=str,
        default=",".join(TARGET_LANGUAGES),
        help="Comma-separated target languages",
    )
    parser.add_argument(
        "--operations",
        type=str,
        default=",".join(OPERATIONS),
        help="Comma-separated operations",
    )
    parser.add_argument(
        "--projections",
        type=str,
        default=",".join(PROJECTIONS),
        help="Comma-separated projections",
    )
    return parser.parse_args()


def find_completions(
    completions_dir: Path,
    model: str,
    target_lang: str,
    operation: str,
    projection: str,
) -> Optional[Path]:
    """Find completion file for a given configuration."""
    completion_path = (
        completions_dir / model / target_lang / operation / projection / DEFAULT_COMPLETION_FILENAME
    )
    if completion_path.exists():
        return completion_path
    return None


def load_completions(completion_file: Path) -> List[Dict]:
    """Load completions from JSON file."""
    with open(completion_file, 'r') as f:
        data = json.load(f)
    
    # Handle wrapped format
    if isinstance(data, dict):
        return data.get("completions", [])
    return data


def evaluate_completions(
    completions: List[Dict],
    wildguard: WildGuardEvaluator,
    gemma3: object,
    target_lang: str,
) -> Tuple[float, float]:
    """Evaluate completions with WildGuard and Gemma-3, return ASR rates."""
    
    # WildGuard
    wg_eval = wildguard.evaluate_all(
        completions=[dict(c) for c in completions],
        logger=None,
        cfg=type("Cfg", (), {"lang": target_lang})(),
    )
    wg_unsafe = sum(int(item.get("wildguard", {}).get("response_harmful", 0)) for item in wg_eval)
    wg_asr = (wg_unsafe / len(wg_eval)) * 100 if wg_eval else 0.0
    
    # Gemma-3
    response_field = "response_translated" if completions[0].get("response_translated") else (
        "response_text" if completions[0].get("response_text") else "response"
    )
    g3_eval = gemma3.evaluate_completions([dict(c) for c in completions], use_field=response_field)
    g3_asr = g3_eval["gemma_3_27b_it_unsafe_rate"] * 100
    
    return wg_asr, g3_asr


def main() -> None:
    args = parse_args()
    
    completions_dir = Path(args.completions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    target_langs = [l.strip() for l in args.target_langs.split(",") if l.strip()]
    operations = [o.strip() for o in args.operations.split(",") if o.strip()]
    projections = [p.strip() for p in args.projections.split(",") if p.strip()]
    
    print(f"\n{'='*80}")
    print(f"REFUSAL VECTOR EVALUATION")
    print(f"{'='*80}")
    print(f"Completions dir: {completions_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Models:          {models}")
    print(f"Target langs:    {target_langs}")
    print(f"Operations:      {operations}")
    print(f"Projections:     {projections}")
    print(f"{'='*80}\n")
    
    # Initialize evaluators
    print("Loading evaluators...")
    wildguard = WildGuardEvaluator()
    gemma3 = create_gemma_3_evaluator()
    print("✅ Evaluators loaded\n")
    
    # Collect all results
    results = {
        "wildguard": {},
        "gemma_3_27b_it": {},
    }
    rows = []
    
    total_start = time.time()
    processed = 0
    missing = 0
    
    for model in models:
        results["wildguard"][model] = {}
        results["gemma_3_27b_it"][model] = {}
        
        print(f"\n[MODEL] {model.upper()}")
        
        for target_lang in target_langs:
            results["wildguard"][model][target_lang] = {}
            results["gemma_3_27b_it"][model][target_lang] = {}
            
            for operation in operations:
                for projection in projections:
                    completion_file = find_completions(
                        completions_dir, model, target_lang, operation, projection
                    )
                    
                    if not completion_file:
                        print(f"  ⊘ {target_lang}/{operation}/{projection} (not found)")
                        missing += 1
                        continue
                    
                    print(f"  ⟳ {target_lang}/{operation}/{projection}")
                    
                    try:
                        eval_start = time.time()
                        
                        # Load and evaluate
                        completions = load_completions(completion_file)
                        wg_asr, g3_asr = evaluate_completions(
                            completions, wildguard, gemma3, target_lang
                        )
                        
                        # Store results
                        key = f"{operation}_{projection}"
                        results["wildguard"][model][target_lang][key] = {
                            "asr": wg_asr,
                            "count": len(completions),
                            "source_path": str(completion_file),
                        }
                        results["gemma_3_27b_it"][model][target_lang][key] = {
                            "asr": g3_asr,
                            "count": len(completions),
                            "source_path": str(completion_file),
                        }
                        
                        # Add to rows
                        rows.append({
                            "Model": model,
                            "Target Lang": target_lang.upper(),
                            "Operation": operation,
                            "Projection": projection,
                            "WildGuard ASR (%)": wg_asr,
                            "Gemma-3 ASR (%)": g3_asr,
                            "Count": len(completions),
                        })
                        
                        elapsed = time.time() - eval_start
                        print(f"    ✅ WG: {wg_asr:.1f}% | G3: {g3_asr:.1f}% ({elapsed:.1f}s)")
                        
                        processed += 1
                        
                    except Exception as e:
                        print(f"    ❌ ERROR: {e}")
    
    # Save results
    results_file = output_dir / "refusal_vectors_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison tables
    if rows:
        df = pd.DataFrame(rows)
        
        # Save CSV
        csv_file = output_dir / "refusal_vectors_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        # Create pivot tables for easy comparison
        wg_pivot = df.pivot_table(
            values="WildGuard ASR (%)",
            index=["Model", "Target Lang"],
            columns=["Operation", "Projection"],
        )
        g3_pivot = df.pivot_table(
            values="Gemma-3 ASR (%)",
            index=["Model", "Target Lang"],
            columns=["Operation", "Projection"],
        )
        
        # Save pivots
        wg_pivot.to_csv(output_dir / "wildguard_asr_pivot.csv")
        g3_pivot.to_csv(output_dir / "gemma3_asr_pivot.csv")
        
        print(f"\n📊 WILDGUARD ASR (%) BY MODEL AND CONFIGURATION:")
        print(wg_pivot.to_string())
        
        print(f"\n📊 GEMMA-3 ASR (%) BY MODEL AND CONFIGURATION:")
        print(g3_pivot.to_string())
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Processed: {processed}")
    print(f"Missing:   {missing}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   - refusal_vectors_results.json")
    print(f"   - refusal_vectors_comparison.csv")
    print(f"   - wildguard_asr_pivot.csv")
    print(f"   - gemma3_asr_pivot.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
