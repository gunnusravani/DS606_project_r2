"""
Smoke test: Generate a few samples from each model/language to verify generation works.

This script generates just 2-3 samples per model/language pair to verify:
1. Models load and can generate responses
2. Responses are stored in the 'response' column
3. Translation works and 'response_translated' is populated for hi/bn
4. Output format is correct before running full generation

Usage:
    python scripts/test_generation_smoke.py --batch-size 2 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base

MODELS: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

LANGUAGES = ("en", "hi", "bn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: generate a few samples to verify generation pipeline works"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help="Comma-separated model keys to test",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(LANGUAGES),
        help="Comma-separated language codes to test (en,hi,bn)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of samples to generate per model/language pair",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Generation batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/asr_smoke_test",
        help="Where to save smoke test outputs",
    )
    return parser.parse_args()


def test_generation_for_pair(
    model_name: str,
    model_path: str,
    lang: str,
    sample_size: int,
    batch_size: int,
    max_new_tokens: int,
    output_dir: Path,
) -> None:
    """Generate a small sample from a model/language pair and inspect output."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name.upper()} / {lang.upper()}")
    print(f"{'='*70}")

    try:
        # Load dataset
        dataset = load_dataset_split(harmtype="harmful", split="test", lang=lang)
        dataset = dataset[:sample_size]
        print(f"✓ Loaded {len(dataset)} test samples for {lang.upper()}")

        # Load model and generate
        print(f"Loading model: {model_path}...")
        model_base = construct_model_base(model_path, lang)
        print(f"✓ Model loaded")

        print(f"Generating {len(dataset)} completions (batch_size={batch_size})...")
        completions = model_base.generate_completions(
            dataset=dataset,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            translation=(lang != "en"),
        )
        print(f"✓ Generated {len(completions)} completions")

        # Inspect first completion
        if completions:
            first = completions[0]
            print(f"\n  First completion keys: {list(first.keys())}")
            print(f"  Instruction: {first.get('instruction', 'N/A')[:80]}...")
            print(f"  Response length: {len(first.get('response', ''))} chars")
            if first.get("response_translated"):
                print(f"  Response (translated) length: {len(first.get('response_translated', ''))} chars")
                print(f"  ✓ response_translated field present")
            else:
                print(f"  ✗ response_translated field missing (expected for {lang})")

        # Save to output
        out_dir = Path(output_dir) / model_name / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "smoke_test_sample.json"
        with open(out_file, "w") as f:
            json.dump(completions, f, indent=2)
        print(f"✓ Saved to: {out_file}")

        # Clean up model
        try:
            model_base.del_model()
        except Exception:
            pass
        print(f"✓ Model cleaned up")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    args = parse_args()

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if "all" in requested_models:
        requested_models = list(MODELS.keys())

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    output_dir = Path(args.output_dir)

    print(f"\n{'='*70}")
    print(f"SMOKE TEST: Generation Pipeline Verification")
    print(f"{'='*70}")
    print(f"Models: {requested_models}")
    print(f"Languages: {langs}")
    print(f"Samples per pair: {args.sample_size}")
    print(f"Output dir: {output_dir}")

    for model_name in requested_models:
        if model_name not in MODELS:
            print(f"✗ Unknown model key: {model_name} - skipping")
            continue

        model_path = MODELS[model_name]
        for lang in langs:
            if lang not in LANGUAGES:
                print(f"✗ Unknown language: {lang} - skipping")
                continue

            test_generation_for_pair(
                model_name=model_name,
                model_path=model_path,
                lang=lang,
                sample_size=args.sample_size,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                output_dir=output_dir,
            )

    print(f"\n{'='*70}")
    print(f"Smoke test complete!")
    print(f"Check outputs in: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
