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

set -e

# Activate conda environment
source /users/student/prjstu/sravani.gunnu/miniconda3/etc/profile.d/conda.sh
conda activate ds606

# Navigate to project directory
cd /users/student/prjstu/sravani.gunnu/DS606_project_r2

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

mkdir -p logs

echo "=========================================="
echo "Starting Llama Guard 4 Evaluation"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check PyTorch version
echo "Checking PyTorch and dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" || {
    echo "PyTorch issue detected, attempting fix..."
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Install/update transformers with Llama Guard support
echo "Installing Llama Guard 4 transformers..."
pip install --upgrade git+https://github.com/huggingface/transformers@v4.51.3-LlamaGuard-preview --no-deps

# Run simple evaluation
echo ""
echo "=========================================="
echo "Running Llama Guard 4 evaluation..."
echo "=========================================="
python scripts/simple_llama_guard_4_eval.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully!"
else
    echo ""
    echo "✗ Evaluation FAILED!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="


