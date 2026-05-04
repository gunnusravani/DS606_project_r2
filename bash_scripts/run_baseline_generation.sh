#!/bin/bash
#SBATCH --job-name=ref_base
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

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Phase 1: BASELINE (NO REFUSAL VECTORS)"
echo "=========================================="
echo ""

# Step 1: Smoke test
echo "[1/3] Running smoke test (3 samples per model/language)..."
python scripts/test_generation_smoke.py --sample-size 3 --batch-size 2 --max-new-tokens 128

if [ $? -eq 0 ]; then
    echo "✓ Smoke test PASSED"
else
    echo "✗ Smoke test FAILED"
    exit 1
fi

echo ""
echo "[2/3] Running full generation (all model×language pairs)..."
python scripts/generate_asr_completions.py --output-dir output --batch-size 8 --max-new-tokens 256 --overwrite

if [ $? -eq 0 ]; then
    echo "✓ Full generation PASSED"
else
    echo "✗ Full generation FAILED"
    exit 1
fi

echo ""
echo "[3/3] Running baseline evaluation (WildGuard + Gemma-3)..."
python scripts/evaluate_hi_bn_en_asr.py --output-dir output/asr_hi_bn_en_step1

if [ $? -eq 0 ]; then
    echo "✓ Baseline evaluation PASSED"
else
    echo "✗ Baseline evaluation FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ BASELINE GENERATION COMPLETE!"
echo "=========================================="
echo ""
echo "Baseline results saved to:"
echo "  - output/asr_hi_bn_en_step1/asr_results.json"
echo "  - output/asr_hi_bn_en_step1/asr_summary.csv"
echo ""
echo "Next steps:"
echo "  1. Add refusal vector logic to model files"
echo "  2. Run: bash bash_scripts/run_postvector_generation.sh"
echo ""
