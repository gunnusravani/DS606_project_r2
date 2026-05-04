"""
Generate ASR completions for specified models and languages.

Saves completions to `output/<model_name>/<lang>/harmful_harm_ablation_evaluations.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base

MODELS: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

LANGUAGES = ("en", "hi", "bn")

DEFAULT_COMPLETION_FILENAME = "harmful_harm_ablation_evaluations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASR completions for models/languages")
    parser.add_argument("--models", type=str, default=",".join(MODELS.keys()), help="Comma-separated model keys to generate (or 'all')")
    parser.add_argument("--languages", type=str, default=",".join(LANGUAGES), help="Comma-separated language codes to generate (en,hi,bn)")
    parser.add_argument("--output-dir", type=str, default="output/asr_generated", help="Where to write generated completions")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional sample size per language; 0 uses full dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completion files")
    return parser.parse_args()


def _select_dataset(lang: str, sample_size: int = 0):
    dataset = load_dataset_split(harmtype="harmful", split="test", lang=lang)
    if sample_size and sample_size > 0:
        dataset = dataset[:sample_size]
    return dataset


def generate_for_pair(model_name: str, model_path: str, lang: str, out_dir: Path, batch_size: int, max_new_tokens: int, sample_size: int, overwrite: bool) -> None:
    dest_dir = out_dir / model_name / lang
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / DEFAULT_COMPLETION_FILENAME
    if dest_file.exists() and not overwrite:
        print(f"Skipping existing: {dest_file} (use --overwrite to replace)")
        return

    print(f"Generating: {model_name} / {lang.upper()} -> {dest_file}")
    model_base = construct_model_base(model_path, lang)
    try:
        dataset = _select_dataset(lang, sample_size=sample_size)
        completions = model_base.generate_completions(
            dataset=dataset,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            translation=(lang != "en"),
        )
        with open(dest_file, "w") as f:
            json.dump(completions, f, indent=2)
        print(f"Saved {len(completions)} completions to {dest_file}")
    finally:
        try:
            model_base.del_model()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    if "all" in requested:
        requested = list(MODELS.keys())

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    out_dir = Path(args.output_dir)

    for m in requested:
        if m not in MODELS:
            print(f"Unknown model key: {m} - skipping")
            continue
        for lang in langs:
            if lang not in LANGUAGES:
                print(f"Unknown language: {lang} - skipping")
                continue
            generate_for_pair(
                model_name=m,
                model_path=MODELS[m],
                lang=lang,
                out_dir=out_dir,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                sample_size=args.sample_size,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
