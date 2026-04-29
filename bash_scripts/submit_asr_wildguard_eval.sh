#!/bin/bash

#SBATCH --job-name=asr_wildguard_eval
#SBATCH --account=irohs_proj2
#SBATCH --partition=cn5_water
#SBATCH --qos=water
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# Activate conda environment
source /users/student/prjstu/sravani.gunnu/miniconda3/etc/profile.d/conda.sh
conda activate ds606

# Navigate to project directory
cd /users/student/prjstu/sravani.gunnu/DS606_project_r2

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

mkdir -p logs output

echo "=========================================="
echo "Comprehensive ASR Evaluation Pipeline"
echo "=========================================="
echo "Time: $(date)"
echo ""
echo "Pipeline Steps:"
echo "  1. Generate responses (Instruct models)"
echo "  2. Translate to English (GoogleTranslator for Hi/Bn)"
echo "  3. Classify with Gemma-3-27B-IT"
echo "  4. Classify with WildGuard"
echo "  5. Compare results"
echo ""
echo "Base Models:"
echo "  - Llama3.1-8B-Instruct"
echo "  - Qwen2.5-7B-Instruct"
echo "  - Gemma2-9B-IT"
echo ""
echo "Languages: English, Hindi, Bengali"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q deep-translator transformers torch

echo "Starting evaluation..."
echo "=========================================="
echo ""

python scripts/evaluate_models_wildguard.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully!"
    echo "Results saved to: output/asr_comprehensive_evaluation.json"
    echo "Responses saved to: output/responses/"
else
    echo ""
    echo "✗ Evaluation FAILED!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
