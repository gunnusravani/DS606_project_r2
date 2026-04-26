#!/usr/bin/env python3
"""
Helper script to update all model files with Hindi and Bengali refusal tokens.

Usage (Option 1 - Auto-load from file):
    python scripts/update_refusal_tokens.py
    
Usage (Option 2 - Manual override):
    python scripts/update_refusal_tokens.py --hi [HINDI_TOKEN_IDS] --bn [BENGALI_TOKEN_IDS]
    
Example:
    python scripts/update_refusal_tokens.py --hi "127748,128976" --bn "12345,67890"
"""

import argparse
import re
import json
from pathlib import Path
from typing import List, Tuple

def parse_tokens(token_str: str) -> List[int]:
    """Parse comma-separated token IDs."""
    return [int(t.strip()) for t in token_str.split(',')]

def load_tokens_from_file(config_dir: Path = None) -> Tuple[List[int], List[int], bool]:
    """
    Load refusal tokens from the saved JSON file.
    
    Returns:
        (hi_tokens, bn_tokens, success)
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / 'config'
    
    tokens_file = config_dir / 'refusal_tokens_hi_bn.json'
    
    if tokens_file.exists():
        try:
            with open(tokens_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            hi_tokens = data.get('hi', [])
            bn_tokens = data.get('bn', [])
            
            if hi_tokens and bn_tokens:
                print(f"✅ Loaded tokens from: {tokens_file}")
                print(f"   Hindi tokens: {hi_tokens}")
                print(f"   Bengali tokens: {bn_tokens}\n")
                return hi_tokens, bn_tokens, True
        except Exception as e:
            print(f"⚠️  Error reading {tokens_file}: {e}")
    
    return [], [], False

def update_model_file(filepath: Path, hi_tokens: List[int], bn_tokens: List[int]) -> bool:
    """Update a model file with Hindi and Bengali refusal tokens."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file has REFUSAL_TOKENS_LANG
        if 'REFUSAL_TOKENS_LANG' not in content:
            print(f"⚠️  {filepath.name}: No REFUSAL_TOKENS_LANG found, skipping")
            return False
        
        # Find the dictionary and add tokens if not already present
        original_content = content
        
        # Pattern to find the closing brace of REFUSAL_TOKENS_LANG
        pattern = r"(REFUSAL_TOKENS_LANG\s*=\s*\{[^}]*?)(\n\s*\})"
        
        def replacement(match):
            dict_content = match.group(1)
            closing_brace = match.group(2)
            
            # Check if Hindi and Bengali already exist
            if "'hi'" not in dict_content:
                dict_content += f",\n    'hi': {hi_tokens},"
            if "'bn'" not in dict_content:
                dict_content += f"\n    'bn': {bn_tokens},"
            
            return dict_content + closing_brace
        
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {filepath.name}: Updated with Hindi {hi_tokens} and Bengali {bn_tokens}")
            return True
        else:
            print(f"⏭️  {filepath.name}: Hindi and Bengali tokens already present")
            return False
            
    except Exception as e:
        print(f"❌ {filepath.name}: Error - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Update model files with Hindi and Bengali refusal tokens'
    )
    parser.add_argument(
        '--hi',
        type=str,
        default=None,
        help='Hindi refusal token IDs (comma-separated, e.g., "127748,128976"). If not provided, loads from config/refusal_tokens_hi_bn.json'
    )
    parser.add_argument(
        '--bn',
        type=str,
        default=None,
        help='Bengali refusal token IDs (comma-separated, e.g., "12345,67890"). If not provided, loads from config/refusal_tokens_hi_bn.json'
    )
    
    args = parser.parse_args()
    
    # Try to load from file first
    hi_tokens = None
    bn_tokens = None
    
    if args.hi is None or args.bn is None:
        print("🔍 Attempting to load tokens from config/refusal_tokens_hi_bn.json...\n")
        hi_tokens, bn_tokens, success = load_tokens_from_file()
        
        if not success:
            print("❌ Could not load tokens from file. Please provide tokens manually:")
            print("   python scripts/update_refusal_tokens.py --hi [IDS] --bn [IDS]")
            return
    
    # Use command-line arguments if provided (override file)
    if args.hi is not None:
        hi_tokens = parse_tokens(args.hi)
        print(f"Using Hindi tokens from command line: {hi_tokens}")
    
    if args.bn is not None:
        bn_tokens = parse_tokens(args.bn)
        print(f"Using Bengali tokens from command line: {bn_tokens}")
    
    print(f"\nUpdating models with:")
    print(f"  Hindi tokens: {hi_tokens}")
    print(f"  Bengali tokens: {bn_tokens}")
    print()
    
    # Find all model files
    model_dir = Path(__file__).parent.parent / 'pipeline' / 'model_utils'
    model_files = list(model_dir.glob('*_model.py'))
    
    print(f"Found {len(model_files)} model files\n")
    
    updated_count = 0
    for filepath in sorted(model_files):
        if update_model_file(filepath, hi_tokens, bn_tokens):
            updated_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: Updated {updated_count} model files")
    print(f"{'='*60}")
    
    if updated_count > 0:
        print("\n✅ All model files updated successfully!")
        print("\nNext steps:")
        print("1. Verify the changes: git diff pipeline/model_utils/")
        print("2. Run the pipeline for Hindi:")
        print("   python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml")
        print("3. Run the pipeline for Bengali:")
        print("   python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml")
    else:
        print("\n⚠️  No files were updated. Check your token format or file path.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
