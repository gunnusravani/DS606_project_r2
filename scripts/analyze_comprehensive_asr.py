"""
Analyze comprehensive ASR evaluation results (Gemma-3-27B vs WildGuard).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("\n" + "="*80)
print("COMPREHENSIVE ASR EVALUATION ANALYSIS")
print("="*80)

# Load results
results_file = Path("output/asr_comprehensive_evaluation.json")

if not results_file.exists():
    print(f"\n✗ Results file not found: {results_file}")
    print("Please run: sbatch bash_scripts/submit_asr_wildguard_eval.sh")
    exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

# Parse results
data = []
for evaluator in ['gemma3_27b', 'wildguard']:
    evaluator_name = 'Gemma-3-27B' if evaluator == 'gemma3_27b' else 'WildGuard'
    for model_name, lang_data in sorted(results[evaluator].items()):
        for lang, metrics in lang_data.items():
            data.append({
                'Evaluator': evaluator_name,
                'Model': model_name,
                'Language': lang.upper(),
                'Total': metrics['total'],
                'Unsafe Count': metrics['unsafe_count'],
                'ASR (%)': metrics['asr'],
            })

df = pd.DataFrame(data)

# Print summary by evaluator
print("\n📊 RESULTS BY EVALUATOR\n")

for evaluator in ['Gemma-3-27B', 'WildGuard']:
    print(f"\n{evaluator}:")
    print("-" * 100)
    evaluator_df = df[df['Evaluator'] == evaluator]
    
    summary = evaluator_df.pivot_table(
        values='ASR (%)',
        index='Model',
        columns='Language'
    )
    summary['Average'] = summary.mean(axis=1)
    print(summary.to_string())
    print(f"\nOverall Average ASR: {summary['Average'].mean():.2f}%")

# Comparison table
print("\n\n📋 EVALUATOR COMPARISON\n")
print("Model-Language pair performance difference (Gemma-3-27B - WildGuard):")
print("-" * 100)

comparison_data = []
for model in sorted(df['Model'].unique()):
    for lang in sorted(df['Language'].unique()):
        g3_df = df[(df['Evaluator'] == 'Gemma-3-27B') & (df['Model'] == model) & (df['Language'] == lang)]
        wg_df = df[(df['Evaluator'] == 'WildGuard') & (df['Model'] == model) & (df['Language'] == lang)]
        
        if not g3_df.empty and not wg_df.empty:
            g3_asr = g3_df['ASR (%)'].values[0]
            wg_asr = wg_df['ASR (%)'].values[0]
            diff = g3_asr - wg_asr
            
            comparison_data.append({
                'Model': model,
                'Language': lang,
                'Gemma-3-27B': g3_asr,
                'WildGuard': wg_asr,
                'Difference (G3-WG)': diff,
                'Abs Difference': abs(diff),
            })

comp_df = pd.DataFrame(comparison_data)
print(comp_df.to_string(index=False))

# Statistics
print("\n\n📈 STATISTICAL ANALYSIS\n")
print(f"Average ASR difference: {comp_df['Difference (G3-WG)'].mean():.2f}pp")
print(f"Max disagreement: {comp_df['Abs Difference'].max():.2f}pp")
print(f"Min disagreement: {comp_df['Abs Difference'].min():.2f}pp")
print(f"Median disagreement: {comp_df['Abs Difference'].median():.2f}pp")

# Which evaluator is more conservative?
gemma3_mean = df[df['Evaluator'] == 'Gemma-3-27B']['ASR (%)'].mean()
wildguard_mean = df[df['Evaluator'] == 'WildGuard']['ASR (%)'].mean()

print(f"\nAverage ASR (Gemma-3-27B): {gemma3_mean:.2f}%")
print(f"Average ASR (WildGuard):   {wildguard_mean:.2f}%")
if gemma3_mean > wildguard_mean:
    print(f"→ Gemma-3-27B is MORE conservative (higher ASR) by {gemma3_mean - wildguard_mean:.2f}pp")
else:
    print(f"→ WildGuard is MORE conservative (higher ASR) by {wildguard_mean - gemma3_mean:.2f}pp")

# Create visualizations
print("\n\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: ASR comparison by model
ax = axes[0, 0]
models = sorted(df['Model'].unique())
x = range(len(models))
width = 0.35

g3_asrs = []
wg_asrs = []
for model in models:
    g3 = df[(df['Model'] == model) & (df['Evaluator'] == 'Gemma-3-27B')]['ASR (%)'].mean()
    wg = df[(df['Model'] == model) & (df['Evaluator'] == 'WildGuard')]['ASR (%)'].mean()
    g3_asrs.append(g3)
    wg_asrs.append(wg)

ax.bar([i - width/2 for i in x], g3_asrs, width, label='Gemma-3-27B', alpha=0.8)
ax.bar([i + width/2 for i in x], wg_asrs, width, label='WildGuard', alpha=0.8)
ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('ASR (%)', fontweight='bold')
ax.set_title('ASR by Model - Evaluator Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Difference by model
ax = axes[0, 1]
diffs = [g3_asrs[i] - wg_asrs[i] for i in range(len(models))]
colors = ['green' if d > 0 else 'red' for d in diffs]
ax.bar(models, diffs, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('Difference (Gemma-3-27B - WildGuard) in %', fontweight='bold')
ax.set_title('ASR Difference by Model', fontweight='bold')
ax.set_xticklabels(models, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Plot 3: Language-wise comparison
ax = axes[1, 0]
languages = sorted(df['Language'].unique())
x = range(len(languages))

g3_asrs_lang = []
wg_asrs_lang = []
for lang in languages:
    g3 = df[(df['Language'] == lang) & (df['Evaluator'] == 'Gemma-3-27B')]['ASR (%)'].mean()
    wg = df[(df['Language'] == lang) & (df['Evaluator'] == 'WildGuard')]['ASR (%)'].mean()
    g3_asrs_lang.append(g3)
    wg_asrs_lang.append(wg)

ax.bar([i - width/2 for i in x], g3_asrs_lang, width, label='Gemma-3-27B', alpha=0.8)
ax.bar([i + width/2 for i in x], wg_asrs_lang, width, label='WildGuard', alpha=0.8)
ax.set_xlabel('Language', fontweight='bold')
ax.set_ylabel('ASR (%)', fontweight='bold')
ax.set_title('ASR by Language - Evaluator Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(languages)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Heatmap of differences
ax = axes[1, 1]
pivot_diff = comp_df.pivot_table(
    values='Difference (G3-WG)',
    index='Model',
    columns='Language'
)
sns.heatmap(pivot_diff, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=ax, 
            cbar_kws={'label': 'Difference (%)'})
ax.set_title('ASR Difference Heatmap\n(Gemma-3-27B - WildGuard)', fontweight='bold')

plt.tight_layout()
output_plot = Path("output/asr_evaluator_comparison.png")
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved: {output_plot}")

# Save CSVs
output_csv = Path("output/asr_comprehensive_summary.csv")
comp_df.to_csv(output_csv, index=False)
print(f"✓ Summary CSV saved: {output_csv}")

output_detailed_csv = Path("output/asr_comprehensive_detailed.csv")
df.to_csv(output_detailed_csv, index=False)
print(f"✓ Detailed CSV saved: {output_detailed_csv}")

print("\n" + "="*80)
print("✓ Analysis complete!")
print("="*80 + "\n")
