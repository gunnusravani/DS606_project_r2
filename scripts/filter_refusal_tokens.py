#!/usr/bin/env python3
"""
Filter refusal tokens to remove noise and keep only meaningful refusal signals.

This analyzes the token list and removes:
- Formatting noise (spaces, newlines, special characters)
- Cross-lingual tokens (e.g., Chinese in Hindi)
- English tokens mixed in
- Keeps only language-specific meaningful characters
"""

import json
from pathlib import Path
from typing import List, Dict

def analyze_tokens(tokens: List[int], tokenizer) -> Dict:
    """
    Analyze tokens to categorize them.
    """
    analysis = {
        'formatting': [],
        'language_specific': [],
        'special': [],
        'other': []
    }
    
    for token_id in tokens:
        token_str = tokenizer.decode([token_id])
        
        # Check if special token
        if token_str.startswith('<|') and token_str.endswith('|>'):
            analysis['special'].append((token_id, token_str))
        # Check if formatting
        elif token_str.strip() == '' or token_str in ['*', '"', '[', '-', '_', '|']:
            analysis['formatting'].append((token_id, repr(token_str)))
        elif '\n' in token_str or len(token_str.strip()) == 0:
            analysis['formatting'].append((token_id, repr(token_str)))
        # Check if language-specific (Unicode characters)
        elif any(ord(c) > 127 for c in token_str):
            analysis['language_specific'].append((token_id, token_str))
        else:
            analysis['other'].append((token_id, token_str))
    
    return analysis

def filter_meaningful_tokens(tokens: List[int], tokenizer, language: str = 'hi') -> List[int]:
    """
    Filter tokens to keep only meaningful ones for refusal detection.
    Remove formatting noise and cross-lingual tokens.
    """
    meaningful = []
    
    for token_id in tokens:
        token_str = tokenizer.decode([token_id])
        
        # Keep special tokens (they mark refusal boundaries)
        if token_str in ['<|im_start|>', '<|im_end|>']:
            # Optional: uncomment to include
            # meaningful.append(token_id)
            pass
        
        # Keep language-specific characters (main refusal indicators)
        elif any(ord(c) > 127 for c in token_str):
            # Filter out cross-lingual noise
            # For Hindi: keep Devanagari (U+0900–U+097F)
            # For Bengali: keep Bengali (U+0980–U+09FF)
            
            if language == 'hi':
                # Devanagari Unicode range
                if any(0x0900 <= ord(c) <= 0x097F for c in token_str):
                    meaningful.append(token_id)
            elif language == 'bn':
                # Bengali Unicode range
                if any(0x0980 <= ord(c) <= 0x09FF for c in token_str):
                    meaningful.append(token_id)
            else:
                # Generic: keep if non-ASCII
                meaningful.append(token_id)
    
    return meaningful

def main():
    from transformers import AutoTokenizer
    
    print("="*80)
    print("REFUSAL TOKEN FILTERING & ANALYSIS")
    print("="*80)
    
    # Load the saved tokens
    config_file = Path(__file__).parent.parent / 'config' / 'refusal_tokens_hi_bn.json'
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        tokens_data = json.load(f)
    
    hi_tokens = tokens_data.get('hi', [])
    bn_tokens = tokens_data.get('bn', [])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    print("\n" + "="*80)
    print("HINDI TOKEN ANALYSIS")
    print("="*80)
    print(f"\nOriginal tokens (count: {len(hi_tokens)}):")
    print(f"  {hi_tokens}")
    
    # Analyze
    hi_analysis = analyze_tokens(hi_tokens, tokenizer)
    
    print(f"\nBreakdown:")
    print(f"  Formatting/Noise: {len(hi_analysis['formatting'])} tokens")
    for token_id, token_str in hi_analysis['formatting'][:5]:
        print(f"    - {token_id}: {token_str}")
    
    print(f"\n  Language-Specific (Hindi): {len(hi_analysis['language_specific'])} tokens")
    for token_id, token_str in hi_analysis['language_specific']:
        print(f"    - {token_id}: '{token_str}'")
    
    print(f"\n  Special Tokens: {len(hi_analysis['special'])} tokens")
    for token_id, token_str in hi_analysis['special']:
        print(f"    - {token_id}: {token_str}")
    
    print(f"\n  Other: {len(hi_analysis['other'])} tokens")
    for token_id, token_str in hi_analysis['other'][:3]:
        print(f"    - {token_id}: '{token_str}'")
    
    # Filter
    hi_filtered = filter_meaningful_tokens(hi_tokens, tokenizer, language='hi')
    
    print(f"\n✅ FILTERED Hindi tokens (count: {len(hi_filtered)}):")
    print(f"  {hi_filtered}")
    print(f"  Removed {len(hi_tokens) - len(hi_filtered)} noisy tokens")
    
    print("\n" + "="*80)
    print("BENGALI TOKEN ANALYSIS")
    print("="*80)
    print(f"\nOriginal tokens (count: {len(bn_tokens)}):")
    print(f"  {bn_tokens}")
    
    # Analyze
    bn_analysis = analyze_tokens(bn_tokens, tokenizer)
    
    print(f"\nBreakdown:")
    print(f"  Formatting/Noise: {len(bn_analysis['formatting'])} tokens")
    for token_id, token_str in bn_analysis['formatting'][:5]:
        print(f"    - {token_id}: {token_str}")
    
    print(f"\n  Language-Specific (Bengali): {len(bn_analysis['language_specific'])} tokens")
    for token_id, token_str in bn_analysis['language_specific']:
        print(f"    - {token_id}: '{token_str}'")
    
    print(f"\n  Special Tokens: {len(bn_analysis['special'])} tokens")
    for token_id, token_str in bn_analysis['special']:
        print(f"    - {token_id}: {token_str}")
    
    print(f"\n  Other: {len(bn_analysis['other'])} tokens")
    for token_id, token_str in bn_analysis['other'][:3]:
        print(f"    - {token_id}: '{token_str}'")
    
    # Filter
    bn_filtered = filter_meaningful_tokens(bn_tokens, tokenizer, language='bn')
    
    print(f"\n✅ FILTERED Bengali tokens (count: {len(bn_filtered)}):")
    print(f"  {bn_filtered}")
    print(f"  Removed {len(bn_tokens) - len(bn_filtered)} noisy tokens")
    
    # Save filtered tokens
    filtered_data = {
        'hi': hi_filtered,
        'bn': bn_filtered,
        'note': 'Filtered tokens: removed formatting noise and cross-lingual artifacts'
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("✅ Filtered tokens saved back to: {config_file}")
    print(f"{'='*80}")
    
    print(f"\n📌 Next step:")
    print(f"   Update model files with filtered tokens:")
    print(f"   python scripts/update_refusal_tokens.py")

if __name__ == "__main__":
    main()
