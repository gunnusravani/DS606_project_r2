"""
Generate completions with refusal vectors: Addition, Subtraction, and Cross-lingual Projections

Supports:
  - Operations: add (+coeff * vector), subtract (-coeff * vector)
  - Projections: same-language, cross-lingual (hi→bn, bn→hi)
  - Models: llama3.1-8b, qwen2.5-7b, gemma2-9b

Output structure:
  output/refusal_vectors/<model>/<target_lang>/<operation>/<projection>/completions.json

Example outputs:
  - llama3.1-8b / hi / add / same_lang → same-language, refusal vector added
  - llama3.1-8b / bn / add / cross_hi_to_bn → cross-lingual, Bengali generation with Hindi vectors
  - qwen2.5-7b / hi / subtract / same_lang → same-language, refusal vector subtracted
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.model_utils.llama3_model import REFUSAL_TOKENS_LANG as LLAMA3_TOKENS
from pipeline.model_utils.qwen2_model import REFUSAL_TOKENS_LANG as QWEN2_TOKENS
from pipeline.model_utils.gemma2_model import REFUSAL_TOKENS_LANG as GEMMA2_TOKENS
from pipeline.utils.hook_utils import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_activation_subtraction_input_pre_hook,
)


# Model configurations (matching baseline)
MODELS: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

LANGUAGES = ("hi", "bn")
OPERATIONS = ("add", "subtract")

# Map projection names to (source_lang, target_lang, operation)
PROJECTIONS = {
    "same_lang": {"src_lang": None, "tgt_lang": None, "type": "same"},
    "cross_hi_to_bn": {"src_lang": "hi", "tgt_lang": "bn", "type": "cross"},
    "cross_bn_to_hi": {"src_lang": "bn", "tgt_lang": "hi", "type": "cross"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate completions with refusal vectors (add/subtract, same/cross-lingual)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help="Comma-separated model keys (e.g., 'llama3.1-8b,qwen2.5-7b')",
    )
    parser.add_argument(
        "--target-langs",
        type=str,
        default=",".join(LANGUAGES),
        help="Comma-separated target languages for generation (hi,bn)",
    )
    parser.add_argument(
        "--operations",
        type=str,
        default=",".join(OPERATIONS),
        help="Comma-separated operations (add,subtract)",
    )
    parser.add_argument(
        "--projections",
        type=str,
        default="same_lang",
        help="Comma-separated projections (same_lang, cross_hi_to_bn, cross_bn_to_hi)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/refusal_vectors",
        help="Base output directory",
    )
    parser.add_argument(
        "--tokens-file",
        type=str,
        default="config/refusal_tokens_multimodel.json",
        help="Path to refusal tokens JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Generation batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Sample size per language (0 = use all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    return parser.parse_args()


def get_refusal_tokens_from_models() -> Dict[str, Dict[str, List[int]]]:
    """Extract refusal tokens directly from model class definitions."""
    print(f"Loading refusal tokens from model class definitions...")
    
    # Map model keys to their token definitions
    token_map = {
        "llama3.1-8b": LLAMA3_TOKENS,
        "qwen2.5-7b": QWEN2_TOKENS,
        "gemma2-9b": GEMMA2_TOKENS,
    }
    
    refusal_tokens = {}
    for model_key, tokens_lang in token_map.items():
        refusal_tokens[model_key] = tokens_lang
        lang_str = ", ".join(f"{lang}({len(toks)} toks)" for lang, toks in tokens_lang.items())
        print(f"   {model_key:15} → {lang_str}")
    
    return refusal_tokens


def load_refusal_tokens(tokens_file: Path) -> Dict[str, Dict[str, List[int]]]:
    """Load refusal tokens from multimodel JSON file (fallback if it exists)."""
    if not tokens_file.exists():
        print(f"⚠️  Tokens file not found: {tokens_file}")
        print(f"   Using refusal tokens from model class definitions instead.\n")
        return get_refusal_tokens_from_models()
    
    print(f"Loading refusal tokens from {tokens_file}...")
    with open(tokens_file, 'r') as f:
        data = json.load(f)
    
    tokens = data.get("tokens", {})
    if not tokens:
        print(f"⚠️  No tokens found in {tokens_file}")
        print(f"   Using refusal tokens from model class definitions instead.\n")
        return get_refusal_tokens_from_models()
    
    print(f"✅ Loaded tokens for {len(tokens)} models")
    for model_key, langs in tokens.items():
        lang_str = ", ".join(f"{lang}({len(toks)} toks)" for lang, toks in langs.items())
        print(f"   {model_key:15} → {lang_str}")
    
    return tokens


def get_hook_fn(operation: str):
    """Get hook function for operation (add or subtract)."""
    if operation == "add":
        return get_activation_addition_input_pre_hook
    elif operation == "subtract":
        return get_activation_subtraction_input_pre_hook
    else:
        raise ValueError(f"Unknown operation: {operation}")


def generate_with_refusal_vectors(
    model_key: str,
    model_path: str,
    source_lang: str,
    target_lang: str,
    operation: str,
    projection_name: str,
    dataset: List[Dict],
    refusal_tokens: Dict[str, List[int]],
    batch_size: int,
    max_new_tokens: int,
) -> List[Dict]:
    """Generate completions using refusal vectors with hooks."""
    
    print(f"      Loading model {model_key}...")
    model_base = construct_model_base(model_path, lang=target_lang)
    
    # Get refusal tokens for source language
    if source_lang not in refusal_tokens.get(model_key, {}):
        print(f"      ⚠️  No refusal tokens for {model_key}/{source_lang}")
        available_langs = list(refusal_tokens.get(model_key, {}).keys())
        if available_langs:
            source_tokens = refusal_tokens[model_key][available_langs[0]]
            print(f"      → Using first available: {available_langs[0]}")
        else:
            print(f"      ❌ No refusal tokens available for {model_key} at all!")
            raise ValueError(f"No refusal tokens found for model {model_key}")
    else:
        source_tokens = refusal_tokens[model_key][source_lang]
    
    # Use first refusal token
    refusal_token_id = source_tokens[0]
    refusal_vector = model_base.model.model.embed_tokens.weight[refusal_token_id].clone().detach()
    
    print(f"      Using refusal token {refusal_token_id} from {source_lang}")
    print(f"      Operation: {operation} | Projection: {projection_name}")
    
    # Setup hook
    hook_fn_class = get_hook_fn(operation)
    hook_fn = hook_fn_class(vector=refusal_vector, coeff=1.0)
    
    # Get attention layer 10 (middle layer)
    layer = 10
    attn_module = model_base.model_attn_modules[layer]
    fwd_pre_hooks = [(attn_module, hook_fn)]
    
    # Generate
    print(f"      Generating {len(dataset)} completions (batch_size={batch_size})...")
    completions = model_base.generate_completions(
        dataset=dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=[],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        translation=(target_lang != "en"),
    )
    
    model_base.del_model()
    
    return completions


def main() -> None:
    args = parse_args()
    
    # Parse arguments
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    target_langs = [l.strip() for l in args.target_langs.split(",") if l.strip()]
    operations = [op.strip() for op in args.operations.split(",") if op.strip()]
    projections = [p.strip() for p in args.projections.split(",") if p.strip()]
    
    out_dir = Path(args.output_dir)
    tokens_file = Path(args.tokens_file)
    
    # Load tokens (with fallback to model class definitions)
    refusal_tokens = load_refusal_tokens(tokens_file)
    
    print(f"\n{'='*80}")
    print(f"REFUSAL VECTOR GENERATION: Addition, Subtraction, Cross-lingual Projections")
    print(f"{'='*80}")
    print(f"Models:      {requested_models}")
    print(f"Target langs: {target_langs}")
    print(f"Operations:  {operations}")
    print(f"Projections: {projections}")
    print(f"Batch size:  {args.batch_size} | Max tokens: {args.max_new_tokens}")
    print(f"Output dir:  {out_dir}")
    print(f"{'='*80}\n")
    
    total_start = time.time()
    completed = 0
    failed = 0
    
    # Iterate over all combinations
    for model_key in requested_models:
        if model_key not in MODELS:
            print(f"✗ Unknown model: {model_key}")
            continue
        
        model_path = MODELS[model_key]
        print(f"\n[MODEL] {model_key.upper()}")
        
        for target_lang in target_langs:
            # Load target dataset
            dataset = load_dataset_split(harmtype="harmful", split="test", lang=target_lang)
            if args.sample_size and args.sample_size > 0:
                dataset = dataset[:args.sample_size]
            
            for operation in operations:
                for projection_name in projections:
                    projection = PROJECTIONS.get(projection_name)
                    if not projection:
                        print(f"  ✗ Unknown projection: {projection_name}")
                        continue
                    
                    # Determine source language
                    if projection["type"] == "same":
                        source_lang = target_lang
                    else:  # cross
                        source_lang = projection["src_lang"]
                    
                    completed += 1
                    
                    # Create output directory
                    dest_dir = out_dir / model_key / target_lang / operation / projection_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / "harmful_harm_ablation_evaluations.json"
                    
                    if dest_file.exists() and not args.overwrite:
                        print(f"  ⊘ {target_lang.upper()}/{operation}/{projection_name} (exists, use --overwrite)")
                        continue
                    
                    print(f"  ⟳ {target_lang.upper()}/{operation}/{projection_name}")
                    
                    try:
                        pair_start = time.time()
                        completions = generate_with_refusal_vectors(
                            model_key=model_key,
                            model_path=model_path,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            operation=operation,
                            projection_name=projection_name,
                            dataset=dataset,
                            refusal_tokens=refusal_tokens,
                            batch_size=args.batch_size,
                            max_new_tokens=args.max_new_tokens,
                        )
                        
                        # Save completions
                        with open(dest_file, 'w') as f:
                            json.dump(completions, f, indent=2)
                        
                        elapsed = time.time() - pair_start
                        print(f"    ✅ Saved {len(completions)} completions in {elapsed:.1f}s")
                        print(f"       → {dest_file}")
                        
                    except Exception as e:
                        print(f"    ❌ ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        failed += 1
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"✅ GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Completed: {completed - failed} / {completed}")
    print(f"Failed:    {failed}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"Output dir: {out_dir}")
    print(f"{'='*80}\n")
    
    print("📊 Output structure:")
    print("   output/refusal_vectors/<model>/<target_lang>/<operation>/<projection>/completions.json")
    print("\n📈 Next step: Evaluate with WildGuard and Gemma-3")
    print("   python scripts/evaluate_refusal_vectors.py --output-dir output/refusal_vectors_results")


if __name__ == "__main__":
    main()
