#!/usr/bin/env python3
"""
Analyze Hindi and Bengali dataset statistics.
"""

import json
from pathlib import Path

dataset_dir = Path("dataset/splits_multi")

print("\n" + "="*70)
print("DATASET STATISTICS - HINDI & BENGALI")
print("="*70)

for lang, lang_name in [('hi', 'HINDI'), ('bn', 'BENGALI')]:
    print(f"\n\n{'='*70}")
    print(f"{lang_name} (Language Code: {lang})")
    print(f"{'='*70}")
    
    stats = {
        'train': {'harmful': 0, 'harmless': 0},
        'val': {'harmful': 0, 'harmless': 0},
        'test': {'harmful': 0, 'harmless': 0},
    }
    
    # Collect statistics
    for split in ['train', 'val', 'test']:
        for harmtype in ['harmful', 'harmless']:
            file_path = dataset_dir / f"{harmtype}_{split}_translated_{lang}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                stats[split][harmtype] = len(data)
    
    # Print formatted statistics
    print("\nDATASET BREAKDOWN:")
    print("-" * 70)
    print(f"{'Split':<12} {'Harmful':<15} {'Harmless':<15} {'Total':<15}")
    print("-" * 70)
    
    total_harmful = 0
    total_harmless = 0
    total_all = 0
    
    for split in ['train', 'val', 'test']:
        harmful_count = stats[split]['harmful']
        harmless_count = stats[split]['harmless']
        total_count = harmful_count + harmless_count
        
        print(f"{split.upper():<12} {harmful_count:<15} {harmless_count:<15} {total_count:<15}")
        
        total_harmful += harmful_count
        total_harmless += harmless_count
        total_all += total_count
    
    print("-" * 70)
    print(f"{'TOTAL':<12} {total_harmful:<15} {total_harmless:<15} {total_all:<15}")
    print("-" * 70)
    
    # Calculate percentages
    if total_all > 0:
        harmful_pct = (total_harmful / total_all) * 100
        harmless_pct = (total_harmless / total_all) * 100
        print(f"\nClass Distribution:")
        print(f"  Harmful:   {total_harmful:>3} ({harmful_pct:>5.1f}%)")
        print(f"  Harmless:  {total_harmless:>3} ({harmless_pct:>5.1f}%)")
    
    # Show train/val/test split
    print(f"\nTrain/Val/Test Split:")
    if total_all > 0:
        train_total = stats['train']['harmful'] + stats['train']['harmless']
        val_total = stats['val']['harmful'] + stats['val']['harmless']
        test_total = stats['test']['harmful'] + stats['test']['harmless']
        print(f"  Train: {train_total:>3} ({train_total/total_all*100:>5.1f}%)")
        print(f"  Val:   {val_total:>3} ({val_total/total_all*100:>5.1f}%)")
        print(f"  Test:  {test_total:>3} ({test_total/total_all*100:>5.1f}%)")

print("\n" + "="*70)
print("SOURCE: PolyRefuse - Multilingual Harmful/Harmless Prompt Dataset")
print("="*70 + "\n")
