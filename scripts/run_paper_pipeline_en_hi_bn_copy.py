"""
Paper-faithful refusal-direction pipeline for EN/HI/BN.

Implements the core sequence used in the paper:
1) Extract candidate directions with difference-in-means over harmful vs harmless prompts.
2) Select best direction by refusal-score reduction (with KL filtering).
3) Transfer the selected direction across target languages with:
   - ablation (remove refusal behavior; applied across layers)
   - addition (add refusal behavior; applied at selected layer)
4) Evaluate ASR with WildGuard and Gemma-3 and report deltas vs baseline.

Usage:
  python scripts/run_paper_pipeline_en_hi_bn.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pandas as pd
import torch

from dataset.load_dataset import load_dataset_split
from evaluators.gemma_3 import create_gemma_3_evaluator
from evaluators.wildguard import WildGuardEvaluator
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)

MODELS: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

SOURCE_LANGS = ["en", "hi", "bn"]
TARGET_LANGS = ["en", "hi", "bn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-faithful EN/HI/BN refusal-direction pipeline")
    parser.add_argument("--models", type=str, default=",".join(MODELS.keys()))
    parser.add_argument("--source-langs", type=str, default=",".join(SOURCE_LANGS))
    parser.add_argument("--target-langs", type=str, default=",".join(TARGET_LANGS))
    parser.add_argument("--output-dir", type=str, default="output/paper_exact_en_hi_bn")
    parser.add_argument("--baseline-dir", type=str, default="output/asr_baseline_vllm")
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample-size", type=int, default=0, help="Optional size for test generation per target language")
    parser.add_argument("--kl-threshold", type=float, default=0.2)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--addact-coeff", type=float, default=1.0)
    parser.add_argument("--ablation-coeff", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_dataset(items: List[str], n: int, seed: int) -> List[str]:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def selection_paths(base_dir: Path, model_key: str, source_lang: str) -> Tuple[Path, Path, Path]:
    root = base_dir / model_key / f"source_{source_lang}" / "selection"
    return root, root / "selected_direction.pt", root / "selected_direction_metadata.json"


def generation_file(base_dir: Path, model_key: str, source_lang: str, target_lang: str, operation: str) -> Path:
    out_dir = base_dir / model_key / f"source_{source_lang}" / f"target_{target_lang}" / operation
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "harmful_harm_ablation_evaluations.json"


def baseline_file(base_dir: Path, model_key: str, target_lang: str) -> Path:
    out_dir = base_dir / model_key / "baseline" / target_lang
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "harmful_harm_ablation_evaluations.json"


def evaluate_completions(completions: List[Dict], target_lang: str, wildguard: WildGuardEvaluator, gemma3: object) -> Tuple[float, float]:
    wg_eval = wildguard.evaluate_all(
        completions=[dict(c) for c in completions],
        logger=None,
        cfg=type("Cfg", (), {"lang": target_lang})(),
    )
    wg_unsafe = sum(int(item.get("wildguard", {}).get("response_harmful", 0)) for item in wg_eval)
    wg_asr = (wg_unsafe / len(wg_eval)) * 100 if wg_eval else 0.0

    if completions and completions[0].get("response_translated"):
        response_field = "response_translated"
    elif completions and completions[0].get("response_text"):
        response_field = "response_text"
    else:
        response_field = "response"

    g3_eval = gemma3.evaluate_completions([dict(c) for c in completions], use_field=response_field)
    g3_asr = g3_eval["gemma_3_27b_it_unsafe_rate"] * 100

    return wg_asr, g3_asr


def run_selection(args: argparse.Namespace, model_key: str, model_path: str, source_lang: str, out_root: Path) -> None:
    sel_root, dir_file, meta_file = selection_paths(out_root, model_key, source_lang)
    sel_root.mkdir(parents=True, exist_ok=True)

    if dir_file.exists() and meta_file.exists() and not args.overwrite:
        print(f"[SELECT] Skip existing selection: {model_key}/{source_lang}")
        return

    print(f"[SELECT] {model_key} source={source_lang}")
    model_base = construct_model_base(model_path, lang=source_lang)

    harmful_train = load_dataset_split("harmful", "train", lang=source_lang, instructions_only=True)
    harmless_train = load_dataset_split("harmless", "train", lang=source_lang, instructions_only=True)
    harmful_val = load_dataset_split("harmful", "val", lang=source_lang, instructions_only=True)
    harmless_val = load_dataset_split("harmless", "val", lang=source_lang, instructions_only=True)

    harmful_train = sample_dataset(harmful_train, args.n_train, args.seed)
    harmless_train = sample_dataset(harmless_train, args.n_train, args.seed + 1)
    harmful_val = sample_dataset(harmful_val, args.n_val, args.seed + 2)
    harmless_val = sample_dataset(harmless_val, args.n_val, args.seed + 3)

    generate_dir = sel_root / "generate_directions"
    candidate_directions = generate_directions(
        system=None,
        model_base=model_base,
        harmful_instructions=harmful_train,
        harmless_instructions=harmless_train,
        artifact_dir=str(generate_dir),
        batch_size=args.batch_size,
    )

    # Minimal cfg object expected by select_direction.
    cfg = SimpleNamespace(
        start_layer=args.start_layer,
        addact_coeff=args.addact_coeff,
        batch_size=args.batch_size,
    )

    pos, layer, direction = select_direction(
        cfg=cfg,
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        candidate_directions=candidate_directions,
        pair_name=("harmful", "harmless"),
        artifact_dir=str(sel_root / "select_direction"),
        kl_threshold=args.kl_threshold,
        mode="ablation",
        top_n=1,
        batch_size=args.batch_size,
    )

    payload = {
        "direction": direction[0].detach().cpu(),
        "source_lang": source_lang,
        "layer": int(layer[0]),
        "position": int(pos[0]),
        "model_key": model_key,
        "model_path": model_path,
    }
    torch.save(payload, dir_file)

    with open(meta_file, "w") as f:
        json.dump(
            {
                "source_lang": source_lang,
                "layer": int(layer[0]),
                "position": int(pos[0]),
                "model_key": model_key,
                "model_path": model_path,
                "n_train": len(harmful_train),
                "n_val": len(harmful_val),
                "kl_threshold": args.kl_threshold,
            },
            f,
            indent=2,
        )

    model_base.del_model()


def run_baseline_generation(args: argparse.Namespace, model_key: str, model_path: str, target_lang: str, out_root: Path) -> None:
    """Generate plain baseline completions before the vector interventions."""
    out_file = baseline_file(out_root, model_key, target_lang)
    if out_file.exists() and not args.overwrite:
        print(f"[BASELINE] Skip existing baseline: {model_key}/{target_lang}")
        return

    print(f"[BASELINE] {model_key} target={target_lang}")
    model_base = construct_model_base(model_path, lang=target_lang)
    dataset = load_dataset_split("harmful", "test", lang=target_lang)
    if args.sample_size and args.sample_size > 0:
        dataset = dataset[: args.sample_size]

    completions = model_base.generate_completions(
        dataset=dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        translation=(target_lang != "en"),
    )

    with open(out_file, "w") as f:
        json.dump(completions, f, indent=2)
    print(f"[BASELINE] Saved: {out_file}")
    model_base.del_model()


def run_generation(args: argparse.Namespace, model_key: str, model_path: str, source_lang: str, target_lang: str, out_root: Path) -> None:
    _, dir_file, meta_file = selection_paths(out_root, model_key, source_lang)
    if not dir_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Selection outputs missing for {model_key}/{source_lang}: {dir_file}")

    selection = torch.load(dir_file, map_location="cpu")
    direction = selection["direction"]
    selected_layer = int(selection["layer"])

    dataset = load_dataset_split("harmful", "test", lang=target_lang)
    if args.sample_size and args.sample_size > 0:
        dataset = dataset[: args.sample_size]

    model_base = construct_model_base(model_path, lang=target_lang)
    direction = direction.to(model_base.model.device)

    # 0) Baseline generation with no intervention.
    run_baseline_generation(args, model_key, model_path, target_lang, out_root)

    # 1) Paper ablation operation: remove refusal behavior across layers.
    ablation_file = generation_file(out_root, model_key, source_lang, target_lang, "ablation")
    if not ablation_file.exists() or args.overwrite:
        fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(
            model_base=model_base,
            direction=direction,
            start_layer=args.start_layer,
            ablation_coeff=args.ablation_coeff,
        )
        completions = model_base.generate_completions(
            dataset=dataset,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            translation=(target_lang != "en"),
        )
        with open(ablation_file, "w") as f:
            json.dump(completions, f, indent=2)
        print(f"[GEN] Saved ablation: {ablation_file}")

    # 2) Paper addition operation: add refusal behavior at selected layer.
    addition_file = generation_file(out_root, model_key, source_lang, target_lang, "addition")
    if not addition_file.exists() or args.overwrite:
        layer = min(selected_layer, len(model_base.model_block_modules) - 1)
        coeff = torch.tensor(args.addact_coeff, device=model_base.model.device)
        add_hook = get_activation_addition_input_pre_hook(vector=direction, coeff=coeff)
        fwd_pre_hooks = [(model_base.model_block_modules[layer], add_hook)]

        completions = model_base.generate_completions(
            dataset=dataset,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=[],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            translation=(target_lang != "en"),
        )
        with open(addition_file, "w") as f:
            json.dump(completions, f, indent=2)
        print(f"[GEN] Saved addition: {addition_file}")

    model_base.del_model()


def run_evaluation(args: argparse.Namespace, model_keys: List[str], source_langs: List[str], target_langs: List[str], out_root: Path) -> None:
    print("[EVAL] Loading evaluators...")
    wildguard = WildGuardEvaluator()
    gemma3 = create_gemma_3_evaluator()

    baseline_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
    rows: List[Dict] = []

    for model_key in model_keys:
        for target_lang in target_langs:
            baseline_file = Path(args.baseline_dir) / model_key / target_lang / "harmful_harm_ablation_evaluations.json"
            if baseline_file.exists():
                with open(baseline_file, "r") as f:
                    baseline_completions = json.load(f)
                baseline_cache[(model_key, target_lang)] = evaluate_completions(
                    baseline_completions,
                    target_lang,
                    wildguard,
                    gemma3,
                )

        for source_lang in source_langs:
            for target_lang in target_langs:
                for operation in ["ablation", "addition"]:
                    comp_file = generation_file(out_root, model_key, source_lang, target_lang, operation)
                    if not comp_file.exists():
                        continue
                    with open(comp_file, "r") as f:
                        completions = json.load(f)
                    wg_asr, g3_asr = evaluate_completions(completions, target_lang, wildguard, gemma3)

                    base = baseline_cache.get((model_key, target_lang))
                    row = {
                        "model": model_key,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "operation": operation,
                        "wildguard_asr": wg_asr,
                        "gemma3_asr": g3_asr,
                    }
                    if base is not None:
                        row["wildguard_baseline_asr"] = base[0]
                        row["gemma3_baseline_asr"] = base[1]
                        row["wildguard_delta_vs_baseline"] = wg_asr - base[0]
                        row["gemma3_delta_vs_baseline"] = g3_asr - base[1]
                    rows.append(row)

    if not rows:
        print("[EVAL] No completions found to evaluate.")
        return

    eval_dir = out_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_json = eval_dir / "paper_exact_en_hi_bn_results.json"
    summary_csv = eval_dir / "paper_exact_en_hi_bn_results.csv"

    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[EVAL] Saved: {summary_json}")
    print(f"[EVAL] Saved: {summary_csv}")


def main() -> None:
    args = parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    source_langs = [s.strip() for s in args.source_langs.split(",") if s.strip()]
    target_langs = [t.strip() for t in args.target_langs.split(",") if t.strip()]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("PAPER-FAITHFUL REFUSAL PIPELINE (EN/HI/BN)")
    print("=" * 88)
    print(f"Models:       {model_keys}")
    print(f"Source langs: {source_langs}")
    print(f"Target langs: {target_langs}")
    print(f"Output dir:   {out_root}")
    print(f"n_train={args.n_train}, n_val={args.n_val}, batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}")
    print("=" * 88)

    start = time.time()

    if not args.skip_selection:
        for model_key in model_keys:
            if model_key not in MODELS:
                print(f"[WARN] Unknown model key: {model_key}, skipping")
                continue
            model_path = MODELS[model_key]
            for source_lang in source_langs:
                run_selection(args, model_key, model_path, source_lang, out_root)

    if not args.skip_generation:
        for model_key in model_keys:
            if model_key not in MODELS:
                continue
            model_path = MODELS[model_key]

            for target_lang in target_langs:
                run_baseline_generation(args, model_key, model_path, target_lang, out_root)

            for source_lang in source_langs:
                for target_lang in target_langs:
                    print(f"[GEN] {model_key}: {source_lang} -> {target_lang}")
                    run_generation(args, model_key, model_path, source_lang, target_lang, out_root)

    if not args.skip_evaluation:
        run_evaluation(args, model_keys, source_langs, target_langs, out_root)

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} minutes ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
