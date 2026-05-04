#!/bin/bash
#SBATCH --job-name=ref_postvec
#SBATCH --account=irohs_proj2
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e  # Exit on any error

# Initialize conda properly
eval "$(/users/student/prjstu/sravani.gunnu/miniconda3/bin/conda shell.bash hook)"
conda activate ds606

# Ensure required packages are installed
echo "Verifying required packages..."
pip install --upgrade accelerate>=0.28.0 transformers>=4.40.0 torch>=2.0.0 2>&1 | tail -5

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Phase 2: POST-REFUSAL VECTORS"
echo "=========================================="
echo ""

# Step 1: Full generation with modified models
echo "[1/2] Generating completions with refusal vectors..."
python scripts/generate_asr_completions.py --output-dir output/asr_post_refusal_vectors --batch-size 8 --max-new-tokens 256 --overwrite

if [ $? -eq 0 ]; then
    echo "✓ Post-refusal generation PASSED"
else
    echo "✗ Post-refusal generation FAILED"
    exit 1
fi

echo ""
echo "[2/2] Evaluating with refusal vectors..."
python scripts/evaluate_hi_bn_en_asr.py --output-dir output/asr_post_refusal_vectors

if [ $? -eq 0 ]; then
    echo "✓ Post-refusal evaluation PASSED"
else
    echo "✗ Post-refusal evaluation FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ POST-REFUSAL VECTOR EVALUATION COMPLETE!"
echo "=========================================="
echo ""
echo "Post-refusal results saved to:"
echo "  - output/asr_post_refusal_vectors/asr_results.json"
echo "  - output/asr_post_refusal_vectors/asr_summary.csv"
echo ""
echo "BASELINE vs POST-REFUSAL COMPARISON:"
echo "========================================"
echo "Baseline:"
cat output/asr_hi_bn_en_step1/asr_summary.csv
echo ""
echo "Post-Refusal Vectors:"
cat output/asr_post_refusal_vectors/asr_summary.csv
echo ""
