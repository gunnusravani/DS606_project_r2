# DS606 Project - Environment Setup Guide

This document describes how to set up two separate conda environments:
1. **ds606_vllm**: For baseline generation with vLLM (torch 2.3.0)
2. **ds606_refusal**: For refusal-vector analysis (torch 2.4.1)

Keeping them separate avoids dependency conflicts.

---

## Environment 1: vLLM Baseline (ds606_vllm)

### Setup

```bash
# 1. Create the environment
conda create -n ds606_vllm python=3.11 -y
conda activate ds606_vllm

# 2. Install PyTorch for CUDA 12.1
pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# 3. Install vLLM and supporting packages
pip install -r requirements_vllm.txt
```

### Verify Installation

```bash
conda activate ds606_vllm
python -c "
import torch, transformers
from vllm import LLM
print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.version.cuda})')
print(f'✓ Transformers {transformers.__version__}')
print(f'✓ vLLM available')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"
```

### Run Baseline Generation

```bash
# Single GPU
conda activate ds606_vllm
python scripts/generate_baseline_vllm.py \
  --output-dir output/asr_baseline_vllm \
  --batch-size 32 \
  --max-tokens 256 \
  --overwrite

# Multi-GPU (tensor parallelism)
conda activate ds606_vllm
python scripts/generate_baseline_vllm.py \
  --output-dir output/asr_baseline_vllm \
  --batch-size 32 \
  --tensor-parallel-size 2 \
  --max-tokens 256 \
  --overwrite
```

### SLURM Job (vLLM Baseline)

```bash
#!/bin/bash
#SBATCH --job-name=ds606_vllm_baseline
#SBATCH --account=irohs_proj2
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# Initialize conda
eval "$(/users/student/prjstu/sravani.gunnu/miniconda3/bin/conda shell.bash hook)"
conda activate ds606_vllm

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

# Create logs directory
mkdir -p logs

# Run baseline generation
python scripts/generate_baseline_vllm.py \
  --output-dir output/asr_baseline_vllm \
  --batch-size 32 \
  --max-tokens 256 \
  --overwrite
```

---

## Environment 2: Refusal Vector Analysis (ds606_refusal)

### Setup

```bash
# 1. Create the environment
conda create -n ds606_refusal python=3.11 -y
conda activate ds606_refusal

# 2. Install PyTorch for CUDA 12.1 (latest)
pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 3. Install transformers and dependencies (compatible with torch 2.4+)
pip install "transformers==4.44.2" "accelerate==0.31.0" \
            "huggingface-hub==0.23.3" "datasets==2.19.2" \
            "pandas==2.2.2" "tqdm==4.66.4" "python-dotenv==1.0.1" \
            "sentencepiece==0.2.0" "safetensors==0.4.3" \
            "numpy==1.24.4"
```

### Verify Installation

```bash
conda activate ds606_refusal
python -c "
import torch, transformers
print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.version.cuda})')
print(f'✓ Transformers {transformers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"
```

### Run Refusal-Vector Pipeline

```bash
conda activate ds606_refusal

# Step 1: Identify tokens (15-30 min)
python scripts/identify_refusal_tokens_multimodel.py

# Step 2: Generate variants (8-12 hours)
python scripts/generate_with_refusal_vectors_extended.py --overwrite

# Step 3: Evaluate (4-6 hours)
python scripts/evaluate_refusal_vectors.py
```

### SLURM Job (Refusal Vectors)

```bash
#!/bin/bash
#SBATCH --job-name=ds606_refusal_pipeline
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

set -e

# Initialize conda
eval "$(/users/student/prjstu/sravani.gunnu/miniconda3/bin/conda shell.bash hook)"
conda activate ds606_refusal

# Set Python path
export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

# Offline mode (models already cached)
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_SKIP_LOGIN=1

# Create logs directory
mkdir -p logs

echo "Step 1: Identify refusal tokens..."
python scripts/identify_refusal_tokens_multimodel.py

echo "Step 2: Generate refusal-vector variants..."
python scripts/generate_with_refusal_vectors_extended.py --overwrite

echo "Step 3: Evaluate variants..."
python scripts/evaluate_refusal_vectors.py

echo "✓ Pipeline complete!"
```

---

## Key Differences

| Component | vLLM (ds606_vllm) | Refusal-Vector (ds606_refusal) |
|-----------|------------------|-------------------------------|
| PyTorch | 2.3.0 | 2.4.1 |
| vLLM | 0.5.0 | Not needed |
| Transformers | 4.42.4 | 4.44.2 |
| Use case | Fast baseline generation | Activation manipulation & analysis |
| GPU memory | ~20GB per model | ~30-40GB per model |
| Time estimate | 2-4 hours per language | 20+ hours for full pipeline |

---

## Troubleshooting

### vLLM Environment Issues

**Q: `OSError: libtorch_global_deps.so: cannot open shared object file`**
- Your PyTorch installation is broken. Reinstall:
  ```bash
  conda activate ds606_vllm
  pip uninstall -y torch torchvision torchaudio
  pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
  ```

**Q: `vllm requires torch==2.3.0, but you have...`**
- Do not upgrade vLLM in this environment. It's pinned to 0.5.0 for compatibility.
- If you need a newer vLLM, create a separate environment.

### Refusal-Vector Environment Issues

**Q: `Import "torch" could not be resolved`**
- PyTorch is installed but VS Code's Python extension doesn't see it. Run in the correct conda:
  ```bash
  conda activate ds606_refusal
  ```

**Q: `HuggingFace offline mode disabled...`**
- Models should already be cached. If not, set offline flags:
  ```bash
  export HF_LOCAL_FILES_ONLY=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ```

---

## Quick Reference

```bash
# Check current environment
conda info --envs

# Switch to vLLM baseline
conda activate ds606_vllm

# Switch to refusal-vector analysis
conda activate ds606_refusal

# Remove an environment (if needed)
conda remove --name ds606_vllm --all

# Reinstall all packages from requirements
pip install -r requirements_vllm.txt --force-reinstall
```

---

## Contact & Support

If you encounter other issues:
1. Check logs in `logs/` directory
2. Verify conda environment is active: `conda info` should show the correct env
3. Verify dependencies: `pip list | grep -E "torch|vllm|transformers"`
4. For SLURM issues, check `.err` file output
