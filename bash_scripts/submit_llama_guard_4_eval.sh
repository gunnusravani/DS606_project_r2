#!/bin/bash
#SBATCH --job-name=llama_guard_4_eval
#SBATCH --account=irohs_proj2
#SBATCH --partition=cn5_water
#SBATCH --qos=water
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e  # Exit on first error

# Activate conda environment
source /users/student/prjstu/sravani.gunnu/miniconda3/etc/profile.d/conda.sh
conda activate ds606

# Navigate to project directory
cd /users/student/prjstu/sravani.gunnu/DS606_project_r2

# Set Python path to ensure imports work
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Starting Llama Guard 4 Evaluation Pipeline"
echo "=========================================="
echo "Time: $(date)"
echo "GPU Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Check if Llama Guard 4 dependencies are installed
echo "Checking Llama Guard 4 dependencies..."
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; print('✓ Transformers available')" 2>/dev/null || {
    echo "Installing Llama Guard 4 dependencies..."
    pip install --no-cache-dir git+https://github.com/huggingface/transformers@v4.51.3-LlamaGuard-preview
}

# Run the re-evaluation
echo ""
echo "=========================================="
echo "Starting Llama Guard 4 re-evaluation..."
echo "=========================================="
python scripts/re_evaluate_with_llama_guard_4.py

if [ $? -eq 0 ]; then
    echo "✓ Llama Guard 4 re-evaluation completed successfully!"
else
    echo "✗ Llama Guard 4 re-evaluation FAILED!"
    exit 1
fi

# Optional: Run comparison analysis after evaluation completes
echo ""
echo "=========================================="
echo "Generating comparison analysis..."
echo "=========================================="
python scripts/compare_evaluation_methods.py

if [ $? -eq 0 ]; then
    echo "✓ Comparison analysis completed successfully!"
else
    echo "✗ Comparison analysis FAILED!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All evaluations completed successfully!"
echo "Time: $(date)"
echo "=========================================="

