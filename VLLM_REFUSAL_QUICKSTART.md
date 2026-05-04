gi# vLLM + Refusal-Vector Pipeline: Quick Start

## **Two Separate Conda Environments**

Your project now uses **two independent conda environments** to avoid dependency conflicts:

### **Environment 1: ds606_vllm** (for baseline generation with vLLM)
- PyTorch: 2.3.0
- vLLM: 0.5.0
- Purpose: Fast batch generation with vLLM

### **Environment 2: ds606_refusal** (for refusal-vector analysis)
- PyTorch: 2.4.1
- vLLM: NOT needed
- Purpose: Activation manipulation, generation, and evaluation

---

## **One-Time Setup**

### **Step 1: Create vLLM Environment**

```bash
conda create -n ds606_vllm python=3.11 -y
conda activate ds606_vllm

pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

pip install -r requirements_vllm.txt
```

Verify: `pip list | grep -E "torch|vllm" | head -5`

### **Step 2: Create Refusal-Vector Environment**

```bash
conda create -n ds606_refusal python=3.11 -y
conda activate ds606_refusal

pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install "transformers==4.44.2" "accelerate==0.31.0" \
            "huggingface-hub==0.23.3" "datasets==2.19.2" \
            "pandas==2.2.2" "tqdm==4.66.4" "python-dotenv==1.0.1" \
            "sentencepiece==0.2.0" "safetensors==0.4.3" "numpy==1.24.4"
```

Verify: `pip list | grep -E "torch|transformers" | head -5`

---

## **Running Your Pipelines**

### **Pipeline A: vLLM Baseline Generation**

```bash
conda activate ds606_vllm

# Local execution (single GPU)
python scripts/generate_baseline_vllm.py \
  --output-dir output/asr_baseline_vllm \
  --batch-size 32 \
  --max-tokens 256 \
  --overwrite
```

**Submit to SLURM:**
```bash
sbatch bash_scripts/submit_vllm_baseline.sh
```

**Expected output:**
- `output/asr_baseline_vllm/<model>/<language>/harmful_harm_ablation_evaluations.json`
- Files contain 571-572 completions each
- ~2-4 hours total for 3 models × 3 languages

### **Pipeline B: Refusal-Vector Analysis**

```bash
conda activate ds606_refusal

# Set offline mode (models already cached)
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_SKIP_LOGIN=1

# Run 3-step pipeline
python scripts/identify_refusal_tokens_multimodel.py      # 15-30 min
python scripts/generate_with_refusal_vectors_extended.py --overwrite  # 8-12 hours
python scripts/evaluate_refusal_vectors.py                # 4-6 hours
```

**Submit to SLURM:**
```bash
sbatch bash_scripts/submit_refusal_pipeline.sh
```

**Expected outputs:**
- `config/refusal_tokens_multimodel.json` (Step 1)
- `output/refusal_vectors/<model>/<lang>/<op>/<proj>/harmful_harm_ablation_evaluations.json` (Step 2: 36 variants)
- `output/refusal_vectors_evaluation/*.csv` (Step 3: comparison tables)

---

## **File Structure Overview**

```
scripts/
├── vllm_client.py                             [NEW] Modular vLLM client class
├── generate_baseline_vllm.py                  [UPDATED] Uses vllm_client.py
├── identify_refusal_tokens_multimodel.py      [UPDATED] Offline mode support
├── generate_with_refusal_vectors_extended.py  [Existing] 36 variants
└── evaluate_refusal_vectors.py                [Existing] Comparison metrics

requirements_vllm.txt                          [NEW] vLLM environment deps
ENVIRONMENT_SETUP.md                           [NEW] Detailed setup guide
VLLM_REFUSAL_QUICKSTART.md                     [NEW] This file
```

---

## **Models Configured**

**Both pipelines use same 3 models:**
1. `meta-llama/Llama-3.1-8B-Instruct` (Meta)
2. `Qwen/Qwen2.5-7B-Instruct` (Alibaba)
3. `google/gemma-2-9b-it` (Google)

---

## **Troubleshooting**

### **"Cannot find vllm_client"**
- Ensure you're running from the project root: `/users/.../DS606_project_r2/`
- Verify PYTHONPATH: `export PYTHONPATH=$(pwd):$PYTHONPATH`

### **"OSError: libtorch_global_deps.so"**
- Your PyTorch install is broken. Reinstall in ds606_vllm:
  ```bash
  conda activate ds606_vllm
  pip uninstall -y torch torchvision torchaudio
  pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0
  ```

### **"HF_HUB_OFFLINE: Cannot reach huggingface.co"**
- Models should be cached. Ensure offline flags are set in your SLURM script.

### **CUDA out of memory**
- Reduce batch size: `--batch-size 16` instead of 32
- Reduce model length: `--max-model-len 2048`
- Use fewer GPUs: `--tensor-parallel-size 1`

---

## **Environment Activation Cheat Sheet**

```bash
# For vLLM baseline
conda activate ds606_vllm

# For refusal-vector analysis
conda activate ds606_refusal

# Check which env is active
conda info | grep "active env"

# List all envs
conda info --envs

# Remove an env (if needed)
conda remove --name ds606_vllm --all
```

---

## **Key Differences from Previous Setup**

| Before | Now |
|--------|-----|
| Single mixed environment | Two specialized environments |
| vLLM + torch 2.4 conflicts | vLLM in isolated torch 2.3 env |
| Direct LLM() calls | Modular VLLMClient class |
| Manual cache management | Automatic cache setup in VLLMClient |
| No offline mode | Proper offline support added |

---

## **Next Steps**

1. **Create both environments** (one-time, ~10 min)
2. **Test vLLM environment** with a small test run
3. **Submit baseline generation** job to SLURM
4. **While waiting**, submit refusal-vector job (uses different queue)
5. **Analyze results** when both pipelines complete

---

**Need help?** See `ENVIRONMENT_SETUP.md` for detailed troubleshooting and reference commands.
