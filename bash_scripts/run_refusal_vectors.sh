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
conda activate ds606_refusal

# Install vLLM (fast, optimized generation framework)
# echo "Installing vLLM for efficient generation..."
# pip install --quiet vllm>=0.4.0 2>&1 | tail -3
# echo "✓ vLLM installed"

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH
# export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export HF_SKIP_LOGIN=1
# Create logs directory
mkdir -p logs

export HF_HOME="/janaki/common/huggingface/"
export HF_HUB_CACHE="$HF_HOME/hub"

python scripts/identify_refusal_tokens_multimodel.py      # ~15-30 min
python scripts/generate_with_refusal_vectors_extended.py --overwrite  # ~8-12 hours
python scripts/evaluate_refusal_vectors.py                # ~4-6 hours