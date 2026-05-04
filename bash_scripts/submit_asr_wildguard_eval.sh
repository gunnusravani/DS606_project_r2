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
echo "  1. Load or generate completions for English, Hindi, and Bengali"
echo "  2. Classify with Gemma-3-27B-it"
echo "  3. Classify with WildGuard"
echo "  4. Compare results"
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
# Remove incompatible torchvision if present
pip uninstall -q -y torchvision || true
# Install compatible PyTorch and dependencies (without unnecessary torchvision)
pip install -q --upgrade torch==2.4.0
pip install -q deep-translator "transformers>=4.45.0" accelerate

echo "Starting evaluation..."
echo "=========================================="
echo ""

python scripts/evaluate_hi_bn_en_asr.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully!"
    echo "Results saved to: output/asr_hi_bn_en_step1/"
else
    echo ""
    echo "✗ Evaluation FAILED!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
