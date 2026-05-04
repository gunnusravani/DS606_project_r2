#!/bin/bash
#SBATCH --job-name=ref_base_vllm
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

# Install vLLM (fast, optimized generation framework)
# echo "Installing vLLM for efficient generation..."
# pip install --quiet vllm>=0.4.0 2>&1 | tail -3
# echo "✓ vLLM installed"

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

# Create logs directory
mkdir -p logs

echo ""
echo "=========================================="
echo "Phase 1: BASELINE (vLLM Generation)"
echo "=========================================="
echo ""

# Step 1: Generate completions using vLLM
echo "[1/2] Generating completions (vLLM)..."
echo "      3 models × 3 languages (en/hi/bn)"
python scripts/generate_baseline_vllm.py \
    --models llama3.1-8b,qwen2.5-7b,gemma2-9b \
    --languages en,hi,bn \
    --output-dir output/asr_baseline_vllm \
    --batch-size 32 \
    --max-tokens 256 \
    --overwrite

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Generation PASSED"
else
    echo ""
    echo "✗ Generation FAILED"
    exit 1
fi

echo ""
echo "[2/2] Evaluating completions (WildGuard + Gemma-3)..."
python scripts/evaluate_hi_bn_en_asr.py \
    --output-dir output/asr_baseline_results

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation PASSED"
else
    echo ""
    echo "✗ Evaluation FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ BASELINE GENERATION & EVALUATION COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Completions: output/asr_baseline_vllm/<model>/<lang>/"
echo "  - ASR Results (JSON): output/asr_baseline_results/asr_results.json"
echo "  - ASR Summary (CSV): output/asr_baseline_results/asr_summary.csv"
echo ""
echo "To view results:"
echo "  cat output/asr_baseline_results/asr_summary.csv"
echo ""
