"""
Baseline ASR Generation using vLLM Framework

vLLM provides efficient batched generation with KV cache optimization,
making it much faster than standard transformers generation.

This script generates completions for 3 models across en/hi/bn and saves
them in format compatible with evaluate_hi_bn_en_asr.py.

Usage:
    python scripts/generate_baseline_vllm.py --output-dir output/asr_baseline_vllm --batch-size 32 --max-tokens 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from dataset.load_dataset import load_dataset_split
from vllm import LLM, SamplingParams

# Model configurations
MODELS: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

LANGUAGES = ("en", "hi", "bn")

# Chat formatting templates per model
CHAT_TEMPLATES = {
    "llama3.1-8b": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "qwen2.5-7b": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    "gemma2-9b": "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ASR completions using vLLM (fast baseline generation)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help="Comma-separated model keys to generate (or 'all')",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(LANGUAGES),
        help="Comma-separated language codes (en,hi,bn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/asr_baseline_vllm",
        help="Directory to save generated completions",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generation batch size (vLLM handles this efficiently)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional sample size per language; 0 uses full test set",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism for large models (for multi-GPU)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing completion files",
    )
    return parser.parse_args()


def format_prompt(model_key: str, instruction: str) -> str:
    """Format instruction as a chat prompt for the model."""
    template = CHAT_TEMPLATES.get(model_key, "")
    return template.format(instruction=instruction)


def generate_completions_vllm(
    model_key: str,
    model_path: str,
    lang: str,
    dataset: List[Dict],
    batch_size: int,
    max_tokens: int,
    tensor_parallel_size: int,
) -> tuple[List[Dict], float]:
    """Generate completions using vLLM for a model/language pair.
    
    Returns (completions list, elapsed time in seconds).
    """
    print(f"\n    [vLLM] Loading model: {model_path}...")
    load_start = time.time()
    
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        local_files_only=True,
    )
    
    load_time = time.time() - load_start
    print(f"    ✓ Model loaded in {load_time:.1f}s")
    
    # Format prompts
    print(f"    Formatting {len(dataset)} prompts...")
    prompts = [format_prompt(model_key, item["instruction"]) for item in dataset]
    
    # Set generation params
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    
    # Generate with vLLM
    print(f"    Generating with batch_size={batch_size}, max_tokens={max_tokens}...")
    gen_start = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    gen_time = time.time() - gen_start
    
    # Extract responses
    completions = []
    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text.strip()
        completions.append({
            "instruction": dataset[i]["instruction"],
            "response": response_text,
            "instruction_translated": dataset[i].get("instruction_translated", ""),
            "response_translated": "",  # Will be filled if translated
        })
    
    elapsed = time.time() - load_start
    print(f"    ✓ Generated {len(completions)} responses in {gen_time:.1f}s")
    print(f"      (~{gen_time/len(completions):.2f}s per sample)")
    
    # Cleanup
    del llm
    
    return completions, elapsed


def generate_for_pair(
    model_key: str,
    model_path: str,
    lang: str,
    dataset: List[Dict],
    out_dir: Path,
    batch_size: int,
    max_tokens: int,
    tensor_parallel_size: int,
    overwrite: bool,
) -> bool:
    """Generate and save completions for a model/language pair.
    
    Returns True if successful, False otherwise.
    """
    dest_dir = out_dir / model_key / lang
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "harmful_harm_ablation_evaluations.json"
    
    if dest_file.exists() and not overwrite:
        print(f"    Skipping existing file (use --overwrite): {dest_file}")
        return True
    
    try:
        completions, elapsed = generate_completions_vllm(
            model_key=model_key,
            model_path=model_path,
            lang=lang,
            dataset=dataset,
            batch_size=batch_size,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        # Save completions
        with open(dest_file, "w") as f:
            json.dump(completions, f, indent=2)
        
        file_size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"    ✓ Saved {len(completions)} completions ({file_size_mb:.1f} MB)")
        print(f"      File: {dest_file}")
        
        return True
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    args = parse_args()
    
    # Parse requested models and languages
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if "all" in requested_models:
        requested_models = list(MODELS.keys())
    
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    out_dir = Path(args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"BASELINE ASR GENERATION USING vLLM")
    print(f"{'='*80}")
    print(f"Models: {requested_models}")
    print(f"Languages: {langs}")
    print(f"Batch size: {args.batch_size} | Max tokens: {args.max_tokens}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*80}\n")
    
    total_start = time.time()
    completed_pairs = 0
    total_pairs = len(requested_models) * len(langs)
    successful = 0
    failed = 0
    
    for model_key in requested_models:
        if model_key not in MODELS:
            print(f"✗ Unknown model key: {model_key} - skipping")
            continue
        
        model_path = MODELS[model_key]
        print(f"\n[MODEL] {model_key.upper()}")
        print(f"  Path: {model_path}\n")
        
        for lang_idx, lang in enumerate(langs, 1):
            completed_pairs += 1
            print(f"  [{lang_idx}/{len(langs)}] Language: {lang.upper()}")
            
            # Load dataset
            print(f"    Loading test set for {lang.upper()}...")
            dataset = load_dataset_split(harmtype="harmful", split="test", lang=lang)
            if args.sample_size and args.sample_size > 0:
                dataset = dataset[:args.sample_size]
            print(f"    ✓ Loaded {len(dataset)} samples")
            
            # Generate
            success = generate_for_pair(
                model_key=model_key,
                model_path=model_path,
                lang=lang,
                dataset=dataset,
                out_dir=out_dir,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                tensor_parallel_size=args.tensor_parallel_size,
                overwrite=args.overwrite,
            )
            
            if success:
                successful += 1
            else:
                failed += 1
            
            print()
    
    total_time = time.time() - total_start
    
    print(f"{'='*80}")
    print(f"✓ GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"  Completed: {successful}/{total_pairs} pairs")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"  Output directory: {out_dir}")
    print(f"{'='*80}\n")
    
    if successful == total_pairs:
        print(f"Next step: Evaluate with WildGuard + Gemma-3")
        print(f"  python scripts/evaluate_hi_bn_en_asr.py --output-dir output/asr_baseline_results\n")


if __name__ == "__main__":
    main()
