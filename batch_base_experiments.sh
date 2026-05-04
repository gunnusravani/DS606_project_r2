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

source /users/student/prjstu/sravani.gunnu/miniconda3/etc/profile.d/conda.sh

conda init
conda activate ds606
# Set Python path to ensure imports work
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

# Create logs directory if it doesn't exist
mkdir -p logs

pip install --upgrade accelerate>=0.28.0 transformers>=4.40.0 torch>=2.0.0 2>&1 | tail -5
# Ensure transformers is available (force reinstall if corrupted)
# echo "Ensuring clean transformers installation..."
# pip install --force-reinstall --no-cache-dir transformers>=4.40.0 torch>=2.0.0 2>&1 | grep -E "(Successfully|Requirement|ERROR|error)" || true


python scripts/test_generation_smoke.py --sample-size 3 --batch-size 2 --max-new-tokens 128

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully!"
else
    echo ""
    echo "✗ Evaluation FAILED!"
    exit 1
fi



