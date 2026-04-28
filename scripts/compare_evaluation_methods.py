"""
Comprehensive analysis comparing Substring Matching vs Llama Guard 4 evaluation metrics.
Generates comparison visualizations and detailed statistics for both Hindi and Bengali.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationComparator:
    """Compare different evaluation methodologies."""
    
    def __init__(self):
        self.base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
        self.output_path = Path("output/cross_lingual_analysis")
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_original_evaluation(self, lang: str, timestamp: str = "20250519-000604") -> Dict:
        """Load original substring matching evaluation results."""
        results_dir = self.base_path / f"{lang}/{timestamp}/1/completions"
        eval_file = results_dir / "harmful_harm_ablation_evaluations.json"
        
        if not eval_file.exists():
            raise FileNotFoundError(f"Original evaluation not found: {eval_file}")
        
        with open(eval_file, 'r') as f:
            return json.load(f)
    
    def load_llama_guard_4_evaluation(self, lang: str, timestamp: str = "20250519-000604") -> Dict:
        """Load Llama Guard 4 evaluation results."""
        results_dir = self.base_path / f"{lang}/{timestamp}/1/completions"
        eval_file = results_dir / "harmful_harm_ablation_llama_guard_4_evaluation.json"
        
        if not eval_file.exists():
            logger.warning(f"Llama Guard 4 evaluation not found: {eval_file}")
            return None
        
        with open(eval_file, 'r') as f:
            return json.load(f)
    
    def load_per_sample_data(self, lang: str, timestamp: str = "20250519-000604") -> Dict:
        """Load per-sample classification results."""
        results_dir = self.base_path / f"{lang}/{timestamp}/1/completions"
        per_sample_file = results_dir / "harmful_harm_ablation_llama_guard_4_per_sample.json"
        
        if not per_sample_file.exists():
            logger.warning(f"Per-sample data not found: {per_sample_file}")
            return None
        
        with open(per_sample_file, 'r') as f:
            return json.load(f)
    
    def calculate_agreement(self, lang: str, timestamp: str = "20250519-000604") -> Dict:
        """Calculate agreement between evaluation methods."""
        per_sample = self.load_per_sample_data(lang, timestamp)
        
        if per_sample is None:
            return None
        
        substring_results = []
        lg4_results = []
        
        for sample in per_sample:
            substring_results.append(sample.get('original_substring_matching_success', 0))
            lg4_results.append(sample.get('llama_guard_4_is_jailbreak', 0))
        
        agreement = sum(1 for s, l in zip(substring_results, lg4_results) if s == l)
        disagreement = len(substring_results) - agreement
        
        # Breakdown of disagreements
        substring_only = sum(1 for s, l in zip(substring_results, lg4_results) if s == 1 and l == 0)
        lg4_only = sum(1 for s, l in zip(substring_results, lg4_results) if s == 0 and l == 1)
        
        return {
            'total_samples': len(substring_results),
            'agreement_count': agreement,
            'agreement_rate': agreement / len(substring_results),
            'disagreement_count': disagreement,
            'substring_only_count': substring_only,
            'lg4_only_count': lg4_only,
            'substring_asr': np.mean(substring_results),
            'lg4_asr': np.mean(lg4_results),
        }
    
    def generate_comparison_report(self, timestamp: str = "20250519-000604") -> str:
        """Generate comprehensive comparison report."""
        report = "╔" + "═" * 98 + "╗\n"
        report += "║" + " " * 20 + "EVALUATION METHODOLOGY COMPARISON REPORT" + " " * 39 + "║\n"
        report += "║" + " " * 15 + "Substring Matching vs Llama Guard 4 (Qwen2.5-7B-Instruct)" + " " * 32 + "║\n"
        report += "╚" + "═" * 98 + "╝\n\n"
        
        languages = ['hi', 'bn']
        comparison_data = []
        
        for lang in languages:
            lang_name = "Hindi" if lang == "hi" else "Bengali"
            report += f"\n{'─' * 100}\n"
            report += f"{lang_name.upper()} RESULTS\n"
            report += f"{'─' * 100}\n\n"
            
            try:
                original = self.load_original_evaluation(lang, timestamp)
                lg4 = self.load_llama_guard_4_evaluation(lang, timestamp)
                agreement = self.calculate_agreement(lang, timestamp)
                
                original_asr = original.get('substring_matching_success_rate', 0)
                lg4_asr = lg4.get('llama_guard_4_unsafe_rate', 0) if lg4 else None
                
                report += f"Original ASR (Substring Matching): {original_asr*100:.2f}%\n"
                if lg4:
                    report += f"New ASR (Llama Guard 4):          {lg4_asr*100:.2f}%\n"
                    report += f"Difference:                       {(lg4_asr - original_asr)*100:+.2f}%\n\n"
                
                if agreement:
                    report += f"Agreement Analysis:\n"
                    report += f"  • Total Samples:                {agreement['total_samples']}\n"
                    report += f"  • Agreement Rate:               {agreement['agreement_rate']*100:.2f}%\n"
                    report += f"  • Disagreement Rate:            {(1-agreement['agreement_rate'])*100:.2f}%\n\n"
                    report += f"Disagreement Breakdown:\n"
                    report += f"  • Substring Match Only:         {agreement['substring_only_count']} samples\n"
                    report += f"  • Llama Guard 4 Only:           {agreement['lg4_only_count']} samples\n\n"
                
                # Hazard distribution
                if lg4 and 'llama_guard_4_hazard_distribution' in lg4:
                    hazard_dist = lg4['llama_guard_4_hazard_distribution']
                    report += f"Hazard Categories Detected (Llama Guard 4):\n"
                    for cat, count in sorted(hazard_dist.items(), key=lambda x: x[1], reverse=True):
                        if count > 0:
                            report += f"  • {cat}: {count} occurrences\n"
                
                comparison_data.append({
                    'Language': lang_name,
                    'Substring Matching ASR (%)': f"{original_asr*100:.2f}",
                    'Llama Guard 4 ASR (%)': f"{lg4_asr*100:.2f}" if lg4_asr else "N/A",
                    'Difference (%)': f"{(lg4_asr - original_asr)*100:+.2f}" if lg4_asr else "N/A",
                    'Agreement Rate (%)': f"{agreement['agreement_rate']*100:.2f}" if agreement else "N/A",
                })
            
            except Exception as e:
                report += f"ERROR: {str(e)}\n"
        
        report += "\n" + "╔" + "═" * 98 + "╗\n"
        report += "║" + " " * 25 + "SUMMARY COMPARISON TABLE" + " " * 50 + "║\n"
        report += "╚" + "═" * 98 + "╝\n\n"
        
        comparison_df = pd.DataFrame(comparison_data)
        report += comparison_df.to_string(index=False) + "\n\n"
        
        report += "\n" + "─" * 100 + "\n"
        report += "KEY INSIGHTS:\n"
        report += "─" * 100 + "\n"
        report += "1. Substring Matching vs Llama Guard 4:\n"
        report += "   • Substring uses simple pattern matching for refusal indicators\n"
        report += "   • Llama Guard 4 uses sophisticated LLM-based safety classification\n"
        report += "   • Differences indicate varying definitions of 'jailbreak success'\n\n"
        report += "2. Robustness:\n"
        report += "   • Llama Guard 4 understands semantic meaning and context\n"
        report += "   • Better for multilingual evaluation (supports Hindi/Bengali)\n"
        report += "   • More comprehensive hazard taxonomy (14 categories)\n\n"
        report += "3. Use Cases:\n"
        report += "   • Substring Matching: Fast baseline, good for large-scale evaluation\n"
        report += "   • Llama Guard 4: More accurate safety assessment, multilingual\n"
        report += "   • Recommendation: Use both for comprehensive evaluation\n"
        
        return report
    
    def plot_comparison(self, timestamp: str = "20250519-000604"):
        """Create comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Evaluation Methodology Comparison: Substring Matching vs Llama Guard 4', 
                     fontsize=16, fontweight='bold')
        
        languages = ['hi', 'bn']
        lang_labels = ['Hindi', 'Bengali']
        
        substring_asrs = []
        lg4_asrs = []
        agreements = []
        
        for lang in languages:
            try:
                original = self.load_original_evaluation(lang, timestamp)
                lg4 = self.load_llama_guard_4_evaluation(lang, timestamp)
                agreement = self.calculate_agreement(lang, timestamp)
                
                substring_asrs.append(original.get('substring_matching_success_rate', 0) * 100)
                lg4_asrs.append(lg4.get('llama_guard_4_unsafe_rate', 0) * 100 if lg4 else 0)
                agreements.append(agreement.get('agreement_rate', 0) * 100 if agreement else 0)
            except Exception as e:
                logger.warning(f"Error processing {lang}: {e}")
                substring_asrs.append(0)
                lg4_asrs.append(0)
                agreements.append(0)
        
        # Plot 1: ASR Comparison
        x = np.arange(len(lang_labels))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, substring_asrs, width, label='Substring Matching', 
                               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = axes[0, 0].bar(x + width/2, lg4_asrs, width, label='Llama Guard 4',
                               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        axes[0, 0].set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('ASR Comparison Across Languages', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(lang_labels)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_ylim(0, 110)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Difference
        differences = [lg4 - sm for sm, lg4 in zip(substring_asrs, lg4_asrs)]
        colors_diff = ['#E74C3C' if d < 0 else '#27AE60' for d in differences]
        
        bars = axes[0, 1].bar(lang_labels, differences, color=colors_diff, alpha=0.8, 
                             edgecolor='black', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0, 1].set_ylabel('Difference (%)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('ASR Difference (LG4 - Substring)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{diff:+.1f}%', ha='center', va='bottom' if diff >= 0 else 'top',
                           fontsize=10, fontweight='bold')
        
        # Plot 3: Agreement Rate
        bars = axes[1, 0].bar(lang_labels, agreements, color='#9B59B6', alpha=0.8,
                             edgecolor='black', linewidth=1.5)
        axes[1, 0].set_ylabel('Agreement Rate (%)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Evaluation Method Agreement', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylim(0, 110)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, agree in zip(bars, agreements):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{agree:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 4: Summary Statistics
        axes[1, 1].axis('off')
        summary_text = f"""
METHODOLOGY COMPARISON SUMMARY

Substring Matching:
  • Pattern-based refusal detection
  • Fast and lightweight
  • Language-agnostic baseline
  • Limited accuracy

Llama Guard 4:
  • LLM-based safety classification
  • Multilingual support (Hindi, Bengali)
  • 14 hazard categories (MLCommons)
  • Semantic understanding
  • More robust and contextual

For robust evaluation:
→ Use Llama Guard 4 as primary metric
→ Report both metrics for transparency
→ Monitor disagreements for insight
"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        output_file = self.output_path / "evaluation_methodology_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {output_file}")
        plt.close()


def main():
    """Main execution."""
    comparator = EvaluationComparator()
    
    # Generate report
    report = comparator.generate_comparison_report()
    
    # Save report
    report_file = comparator.output_path / "evaluation_methodology_comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to {report_file}")
    
    # Generate plots
    comparator.plot_comparison()
    
    return comparator


if __name__ == "__main__":
    comparator = main()
