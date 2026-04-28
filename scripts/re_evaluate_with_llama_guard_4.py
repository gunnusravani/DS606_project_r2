"""
Re-evaluate Hindi and Bengali ablation results using Llama Guard 4.
Replaces substring matching with more robust LLM-based classification.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from evaluators.llama_guard_4 import create_llama_guard_4_evaluator
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationReevaluator:
    """Re-evaluate ablation results with Llama Guard 4."""
    
    def __init__(self):
        self.evaluator = create_llama_guard_4_evaluator()
        self.base_path = Path("output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct")
    
    def load_ablation_completions(self, lang: str, timestamp: str = "20250519-000604") -> Tuple[list, Path]:
        """
        Load the harmful_harm_ablation_evaluations.json completions.
        
        Args:
            lang: Language code ('hi' or 'bn')
            timestamp: Results timestamp
            
        Returns:
            Tuple of (completions list, results directory path)
        """
        results_dir = self.base_path / f"{lang}/{timestamp}/1/completions"
        completions_file = results_dir / "harmful_harm_ablation_evaluations.json"
        
        logger.info(f"Loading completions from {completions_file}")
        
        if not completions_file.exists():
            raise FileNotFoundError(f"Completions file not found: {completions_file}")
        
        with open(completions_file, 'r') as f:
            data = json.load(f)
        
        completions = data.get("completions", [])
        logger.info(f"Loaded {len(completions)} completions for {lang.upper()}")
        
        return completions, results_dir
    
    def re_evaluate_language(self, lang: str, timestamp: str = "20250519-000604") -> Dict:
        """
        Re-evaluate all completions for a language using Llama Guard 4.
        
        Args:
            lang: Language code ('hi' or 'bn')
            timestamp: Results timestamp
            
        Returns:
            Dictionary with original and new evaluation results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Re-evaluating {lang.upper()} with Llama Guard 4")
        logger.info(f"{'='*70}")
        
        # Load completions
        completions, results_dir = self.load_ablation_completions(lang, timestamp)
        
        # Determine which response field to use
        use_field = "response_translated" if lang != "en" else "response"
        
        # Evaluate with Llama Guard 4
        evaluation = self.evaluator.evaluate_completions(completions, use_field=use_field)
        
        # Print summary
        logger.info(self.evaluator.get_summary_stats(evaluation))
        
        # Save new evaluation
        output_file = results_dir / f"harmful_harm_ablation_llama_guard_4_evaluation.json"
        with open(output_file, 'w') as f:
            # Only save relevant fields, not the full completions
            summary = {
                'llama_guard_4_unsafe_rate': evaluation['llama_guard_4_unsafe_rate'],
                'llama_guard_4_safe_rate': evaluation['llama_guard_4_safe_rate'],
                'llama_guard_4_error_rate': evaluation['llama_guard_4_error_rate'],
                'llama_guard_4_total_classified': evaluation['llama_guard_4_total_classified'],
                'llama_guard_4_hazard_distribution': evaluation['llama_guard_4_hazard_distribution'],
            }
            json.dump(summary, f, indent=4)
        
        logger.info(f"Saved Llama Guard 4 evaluation to {output_file}")
        
        # Also save per-sample classifications
        per_sample_file = results_dir / f"harmful_harm_ablation_llama_guard_4_per_sample.json"
        per_sample_classifications = []
        
        for i, completion in enumerate(evaluation['completions']):
            per_sample_classifications.append({
                'index': i,
                'prompt': completion.get('instruction', ''),
                'response': completion.get(use_field, ''),
                'original_substring_matching_success': completion.get('is_jailbreak_substring_matching'),
                'llama_guard_4_safety_label': completion['llama_guard_4']['safety_label'],
                'llama_guard_4_hazard_categories': completion['llama_guard_4']['hazard_categories'],
                'llama_guard_4_is_jailbreak': completion['llama_guard_4']['is_unsafe'],  # Unsafe = jailbreak success
            })
        
        with open(per_sample_file, 'w') as f:
            json.dump(per_sample_classifications, f, indent=4)
        
        logger.info(f"Saved per-sample classifications to {per_sample_file}")
        
        return evaluation
    
    def compare_evaluation_methods(self, lang: str, timestamp: str = "20250519-000604") -> pd.DataFrame:
        """
        Compare substring matching vs Llama Guard 4 results.
        
        Args:
            lang: Language code
            timestamp: Results timestamp
            
        Returns:
            DataFrame with comparison metrics
        """
        completions, _ = self.load_ablation_completions(lang, timestamp)
        
        # Get substring matching results
        substring_jailbreaks = [c.get('is_jailbreak_substring_matching', 0) for c in completions]
        substring_rate = np.mean(substring_jailbreaks)
        
        # Re-evaluate with Llama Guard 4
        evaluation = self.re_evaluate_language(lang, timestamp)
        
        # Build comparison
        comparison = {
            'Language': lang.upper(),
            'Substring Matching ASR (%)': round(substring_rate * 100, 2),
            'Llama Guard 4 ASR (%)': round(evaluation['llama_guard_4_unsafe_rate'] * 100, 2),
            'Total Samples': len(completions),
            'Substring Matching Jailbreaks': int(np.sum(substring_jailbreaks)),
            'Llama Guard 4 Unsafe': int(evaluation['llama_guard_4_total_classified'] * evaluation['llama_guard_4_unsafe_rate']),
            'Agreement Samples': 0,  # Will calculate below
        }
        
        # Calculate agreement between methods
        lg4_jailbreaks = [c['llama_guard_4']['is_unsafe'] for c in evaluation['completions']]
        agreement = sum(1 for s, l in zip(substring_jailbreaks, lg4_jailbreaks) if s == l)
        comparison['Agreement Samples'] = agreement
        comparison['Agreement Rate (%)'] = round(agreement / len(completions) * 100, 2)
        
        return comparison, evaluation


def main():
    """Main execution."""
    re_evaluator = AblationReevaluator()
    
    # Re-evaluate both languages
    languages = ['hi', 'bn']
    results = {}
    
    for lang in languages:
        try:
            logger.info(f"\nProcessing {lang.upper()}...")
            comparison, evaluation = re_evaluator.compare_evaluation_methods(lang)
            results[lang] = (comparison, evaluation)
        except Exception as e:
            logger.error(f"Error processing {lang}: {str(e)}")
            raise
    
    # Create comparison table
    comparison_df = pd.DataFrame([results[lang][0] for lang in languages])
    
    print("\n" + "="*100)
    print("COMPARISON: Substring Matching vs Llama Guard 4")
    print("="*100)
    print(comparison_df.to_string(index=False))
    print("="*100)
    
    # Save comparison
    output_file = Path("output/cross_lingual_analysis/lg4_vs_substring_matching_comparison.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\nComparison saved to {output_file}")
    
    return results, comparison_df


if __name__ == "__main__":
    results, comparison_df = main()
