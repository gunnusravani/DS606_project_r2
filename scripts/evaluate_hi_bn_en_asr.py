"""
Step 1 ASR evaluation for English, Hindi, and Bengali.

This script loads existing harmful completion artifacts when available, or can
generate them for missing model/language pairs, and then evaluates the responses
with both WildGuard and Gemma-3-27B-it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import time

from dataset.load_dataset import load_dataset_split
from evaluators.gemma_3 import create_gemma_3_evaluator
from evaluators.wildguard import WildGuardEvaluator
from pipeline.model_utils.model_factory import construct_model_base


MODELS: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

LANGUAGES = ("en", "hi", "bn")

DEFAULT_COMPLETION_FILENAME = "harmful_harm_ablation_evaluations.json"

MODEL_PATH_HINTS: Dict[str, List[str]] = {
    "llama3.1-8b": ["Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"],
    "qwen2.5-7b": ["Qwen2.5-7B-Instruct"],
    "gemma2-9b": ["gemma-2-9b-it"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR on English, Hindi, and Bengali.")
    parser.add_argument("--output-dir", type=str, default="output/asr_hi_bn_en_step1", help="Where to write summaries and regenerated completions.")
    parser.add_argument("--generate-missing", action="store_true", help="Generate missing completion files from the base models.")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size when creating missing completions.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens for generation when creating missing completions.")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional sample size per language. 0 means use the full available dataset or completion file.")
    return parser.parse_args()


def _candidate_completion_files(model_name: str, model_path: str, lang: str) -> List[Path]:
    hints = [model_path, model_name] + MODEL_PATH_HINTS.get(model_name, [])
    matches: List[Path] = []
    for path in Path("output").rglob(DEFAULT_COMPLETION_FILENAME):
        path_text = str(path)
        if f"/{lang}/" not in path_text:
            continue
        if any(hint in path_text for hint in hints):
            matches.append(path)
    return sorted(matches)


def load_existing_completions(model_name: str, model_path: str, lang: str) -> Optional[List[Dict]]:
    matches = _candidate_completion_files(model_name, model_path, lang)
    if not matches:
        return None

    completion_path = matches[0]
    with open(completion_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        completions = data.get("completions", [])
    else:
        completions = data

    print(f"Loaded {len(completions)} completions from {completion_path}")
    return completions


def _select_dataset(lang: str, sample_size: int = 0) -> List[Dict]:
    dataset = load_dataset_split(harmtype="harmful", split="test", lang=lang)
    if sample_size and sample_size > 0:
        dataset = dataset[:sample_size]
    return dataset


def generate_completions(model_path: str, lang: str, batch_size: int, max_new_tokens: int, sample_size: int) -> List[Dict]:
    print(f"Generating completions for {model_path} / {lang.upper()}...")
    model_base = construct_model_base(model_path, lang)
    dataset = _select_dataset(lang, sample_size=sample_size)
    completions = model_base.generate_completions(
        dataset=dataset,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        translation=(lang != "en"),
    )
    model_base.del_model()
    return completions


def resolve_completion_set(model_name: str, model_path: str, lang: str, args: argparse.Namespace) -> Tuple[List[Dict], Optional[Path]]:
    completions = load_existing_completions(model_name, model_path, lang)
    if completions is not None:
        return completions, None

    if not args.generate_missing:
        return [], None

    completions = generate_completions(
        model_path=model_path,
        lang=lang,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        sample_size=args.sample_size,
    )

    out_dir = Path(args.output_dir) / model_name / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    completion_path = out_dir / DEFAULT_COMPLETION_FILENAME
    with open(completion_path, "w") as f:
        json.dump(completions, f, indent=2)
    print(f"Saved generated completions to {completion_path}")
    return completions, completion_path


def pick_response_field(completion: Dict) -> str:
    if completion.get("response_translated"):
        return "response_translated"
    return "response"


def evaluate_model_language_pairs(args: argparse.Namespace) -> Tuple[Dict, List[Dict]]:
    wildguard = WildGuardEvaluator()
    gemma3 = create_gemma_3_evaluator()

    results: Dict[str, Dict] = {
        "wildguard": {},
        "gemma_3_27b_it": {},
    }
    rows: List[Dict] = []

    for model_name, model_path in MODELS.items():
        print(f"\n=== {model_name.upper()} ===")
        results["wildguard"][model_name] = {}
        results["gemma_3_27b_it"][model_name] = {}

        for lang in LANGUAGES:
            completions, source_path = resolve_completion_set(model_name, model_path, lang, args)
            if not completions:
                print(f"Skipping {model_name} / {lang.upper()} - no completion file and generation disabled.")
                continue

            response_field = pick_response_field(completions[0])
            print(f"Evaluating {model_name} / {lang.upper()} using field '{response_field}'")

            wildguard_eval = wildguard.evaluate_all(completions=[dict(item) for item in completions], logger=None, cfg=type("Cfg", (), {"lang": lang})())
            wildguard_unsafe_count = sum(int(item.get("wildguard", {}).get("response_harmful", 0)) for item in wildguard_eval)
            wildguard_unsafe_rate = wildguard_unsafe_count / len(wildguard_eval) if wildguard_eval else 0.0

            gemma3_eval = gemma3.evaluate_completions([dict(item) for item in completions], use_field=response_field)
            gemma3_unsafe_rate = gemma3_eval["gemma_3_27b_it_unsafe_rate"]
            gemma3_unsafe_count = sum(item.get("gemma_3_27b_it", {}).get("is_unsafe", 0) for item in gemma3_eval["completions"])

            results["wildguard"][model_name][lang] = {
                "total": len(completions),
                "unsafe_count": wildguard_unsafe_count,
                "asr": wildguard_unsafe_rate * 100,
                "source_path": str(source_path) if source_path else None,
                import time
                total_start = time.time()
                print(f"\n{'='*70}")
                print(f"EVALUATION: WildGuard & Gemma-3-27B-it (Zero-shot)")
                print(f"{'='*70}\n")

                pair_count = 0
                for model_name, model_path in MODELS.items():
                    print(f"\n[MODEL] {model_name.upper()}")
                    print(f"  Path: {model_path}")
                    results["wildguard"][model_name] = {}
                    results["gemma_3_27b_it"][model_name] = {}
            results["gemma_3_27b_it"][model_name][lang] = {
                    for lang_idx, lang in enumerate(LANGUAGES, 1):
                        pair_count += 1
                        print(f"\n  [{lang_idx}/3] Language: {lang.upper()}")
            
                        pair_start = time.time()
                        completions, source_path = resolve_completion_set(model_name, model_path, lang, args)
                        if not completions:
                            print(f"      ✗ Skipping - no completion file and generation disabled")
                            continue
                "total": len(completions),
                        response_field = pick_response_field(completions[0])
                        print(f"      Loaded {len(completions)} completions (field: '{response_field}')")
                        print(f"      Source: {source_path if source_path else 'discovered'}")
                "unsafe_count": gemma3_unsafe_count,
                        print(f"      Evaluating with WildGuard...")
                        wg_start = time.time()
                        wildguard_eval = wildguard.evaluate_all(completions=[dict(item) for item in completions], logger=None, cfg=type("Cfg", (), {"lang": lang})())
                        wg_time = time.time() - wg_start
                        wildguard_unsafe_count = sum(int(item.get("wildguard", {}).get("response_harmful", 0)) for item in wildguard_eval)
                        wildguard_unsafe_rate = wildguard_unsafe_count / len(wildguard_eval) if wildguard_eval else 0.0
                        print(f"        ✓ WildGuard: {wildguard_unsafe_rate*100:.1f}% ASR ({wg_time:.1f}s)")
                "asr": gemma3_unsafe_rate * 100,
                        print(f"      Evaluating with Gemma-3-27B-it...")
                        g3_start = time.time()
                        gemma3_eval = gemma3.evaluate_completions([dict(item) for item in completions], use_field=response_field)
                        g3_time = time.time() - g3_start
                        gemma3_unsafe_rate = gemma3_eval["gemma_3_27b_it_unsafe_rate"]
                        gemma3_unsafe_count = sum(item.get("gemma_3_27b_it", {}).get("is_unsafe", 0) for item in gemma3_eval["completions"])
                        print(f"        ✓ Gemma-3: {gemma3_unsafe_rate*100:.1f}% ASR ({g3_time:.1f}s)")
            
                        pair_elapsed = time.time() - pair_start
                        print(f"      Pair time: {pair_elapsed:.1f}s")

                "source_path": str(source_path) if source_path else None,
            }

            rows.append({
                "Model": model_name,
                "Language": lang.upper(),
                "Evaluator": "WildGuard",
                "Total": len(completions),
                "Unsafe Count": wildguard_unsafe_count,
                "ASR (%)": wildguard_unsafe_rate * 100,
                "Source": str(source_path) if source_path else "generated",
            })
            rows.append({
                "Model": model_name,
                "Language": lang.upper(),
                "Evaluator": "Gemma-3-27B-it",
                "Total": len(completions),
                "Unsafe Count": gemma3_unsafe_count,
                "ASR (%)": gemma3_unsafe_rate * 100,
                "Source": str(source_path) if source_path else "generated",
            })

            print(f"  WildGuard ASR:      {wildguard_unsafe_rate * 100:.2f}% ({wildguard_unsafe_count}/{len(completions)})")
            print(f"  Gemma-3-27B-it ASR: {gemma3_unsafe_rate * 100:.2f}% ({gemma3_unsafe_count}/{len(completions)})")

    return results, rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, rows = evaluate_model_language_pairs(args)

    results_path = out_dir / "asr_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(rows)
    summary_path = out_dir / "asr_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\nSummary table")
    if not df.empty:
        print(df.pivot_table(values="ASR (%)", index=["Model", "Language"], columns="Evaluator").to_string())
    print(f"\nSaved JSON: {results_path}")
    print(f"Saved CSV:  {summary_path}")
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"✓ EVALUATION COMPLETE!")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"{'='*70}\n")
    
    print(f"{'='*70}")
    print(f"ASR RESULTS SUMMARY")
    print(f"{'='*70}")
    if not df.empty:
        pivot = df.pivot_table(values="ASR (%)", index=["Model", "Language"], columns="Evaluator")
        print(pivot.to_string())
    print(f"{'='*70}\n")
    
    print(f"📊 Results saved to:")
    print(f"   JSON: {results_path}")
    print(f"   CSV:  {summary_path}\n")

if __name__ == "__main__":
    main()