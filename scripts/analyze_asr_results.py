"""
Parse and visualize ASR results from WildGuard evaluation.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("\n" + "="*80)
print("ASR RESULTS - CROSS-LINGUAL ANALYSIS (Gemma-3-27B Evaluator)")
print("="*80)

# Load results
results_file = Path("output/asr_gemma3_evaluator_multilingual.json")

if not results_file.exists():
    print(f"\n✗ Results file not found: {results_file}")
    print("Please run: sbatch bash_scripts/submit_asr_wildguard_eval.sh")
    exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

# Create summary DataFrame
data = []
for model_name, lang_data in sorted(results.items()):
    for lang, metrics in lang_data.items():
        data.append({
            'Model': model_name,
            'Language': lang.upper(),
            'Total Prompts': metrics['total'],
            'Unsafe Count': metrics['unsafe_count'],
            'Safe Count': metrics['safe_count'],
            'ASR (%)': metrics['asr'],
            'Safe Rate (%)': metrics['safe_rate'],
        })

df = pd.DataFrame(data)

# Print summary table
print("\n📊 SUMMARY TABLE: Attack Success Rate (%) - Gemma-3-27B-IT Evaluator")
print("-" * 100)

summary_pivot = df.pivot_table(
    values='ASR (%)',
    index='Model',
    columns='Language',
    aggfunc='first'
)
summary_pivot['Average'] = summary_pivot.mean(axis=1)

print(summary_pivot.to_string())
print("-" * 100)

# Print detailed breakdown
print("\n📋 DETAILED BREAKDOWN:")
print("-" * 100)

for model in sorted(df['Model'].unique()):
    model_df = df[df['Model'] == model]
    print(f"\n{model.upper()}:")
    for _, row in model_df.iterrows():
        print(f"  {row['Language']}:")
        print(f"    Total Prompts: {row['Total Prompts']}")
        print(f"    Safe: {row['Safe Count']} ({row['Safe Rate (%)']:.1f}%)")
        print(f"    Unsafe/ASR: {row['Unsafe Count']} ({row['ASR (%)']:.1f}%)")

# Create visualizations
print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: ASR by Model and Language
ax1 = axes[0]
models = df['Model'].unique()
languages = df['Language'].unique()

x_pos = range(len(models))
width = 0.35

for i, lang in enumerate(sorted(languages)):
    lang_df = df[df['Language'] == lang].sort_values('Model')
    values = lang_df['ASR (%)'].values
    ax1.bar([p + i*width for p in x_pos], values, width, label=lang)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('ASR by Model (Gemma-3-27B Evaluator)', fontsize=13, fontweight='bold')
ax1.set_xticks([p + width/2 for p in x_pos])
ax1.set_xticklabels(sorted(models), rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Heatmap
ax2 = axes[1]
heatmap_data = df.pivot_table(
    values='ASR (%)',
    index='Model',
    columns='Language',
    aggfunc='first'
)
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax2, cbar_kws={'label': 'ASR (%)'})
ax2.set_title('ASR Heatmap - Gemma-3-27B Evaluation', fontsize=13, fontweight='bold')

plt.tight_layout()
output_plot = Path("output/asr_comparison_gemma3_evaluator.png")
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved: {output_plot}")

# Save summary as CSV
output_csv = Path("output/asr_summary_gemma3_evaluator.csv")
summary_pivot.to_csv(output_csv)
print(f"✓ Summary CSV saved: {output_csv}")

# Save detailed results as CSV
output_detailed_csv = Path("output/asr_detailed_gemma3_evaluator.csv")
df.to_csv(output_detailed_csv, index=False)
print(f"✓ Detailed CSV saved: {output_detailed_csv}")

print("\n" + "="*80)
print("✓ Analysis complete!")
print("="*80 + "\n")
