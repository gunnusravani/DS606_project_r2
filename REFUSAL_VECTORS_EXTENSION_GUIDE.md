# Refusal Vector Extension: Complete Implementation Guide

## 📋 What Was Implemented

You now have a **complete system** for:
1. ✅ Identifying refusal tokens across **3 models** (Llama-3.1-8B, Qwen2.5-7B, Qwen2-7B)
2. ✅ Generating completions with **both addition and subtraction** of refusal vectors
3. ✅ **Cross-lingual projections** (Hindi vectors on Bengali, and vice versa)
4. ✅ Evaluating **36 unique variants** across WildGuard and Gemma-3-27B-it

---

## 🔧 System Architecture

### Modified/Created Files

**Model Wrappers (Updated)**:
- `pipeline/model_utils/llama2_model.py` → Added REFUSAL_TOKENS_LANG with hi/bn
- `pipeline/model_utils/qwen_model.py` → Added REFUSAL_TOKENS_LANG with hi/bn
- All 8 models now support language-specific refusal tokens

**Hooks (Extended)**:
- `pipeline/utils/hook_utils.py` → Added `get_activation_subtraction_input_pre_hook()`
- Available hooks:
  - `get_activation_addition_input_pre_hook(vector, coeff)` → activation += coeff * vector
  - `get_activation_subtraction_input_pre_hook(vector, coeff)` → activation -= coeff * vector

**Scripts (Created)**:
- `scripts/identify_refusal_tokens_multimodel.py` → Extract tokens from all 3 models
- `scripts/generate_with_refusal_vectors_extended.py` → Generate all 36 variants
- `scripts/evaluate_refusal_vectors.py` → Evaluate all variants

**Configs (Created)**:
- `config/refusal_tokens_multimodel.json` → Stores tokens per model/language (auto-generated)
- `config/cross_lingual_projections.json` → Projection definitions

---

## 📊 Variants Explained

### Models (3)
- `llama3.1-8b` → meta-llama/Llama-3.1-8B-Instruct
- `qwen2.5-7b` → Qwen/Qwen2.5-7B-Instruct
- `gemma2-9b` → google/gemma-2-9b-it

### Target Languages (2)
- `hi` → Hindi
- `bn` → Bengali

### Operations (2)
- `add` → activation += coeff × refusal_vector (strengthen refusal)
- `subtract` → activation -= coeff × refusal_vector (weaken refusal)

### Projections (3)
- `same_lang` → Use source language vectors on same language
  - Example: Hindi vectors on Hindi generation
- `cross_hi_to_bn` → Use Hindi vectors on Bengali generation
  - Example: Hindi refusal vectors applied to Bengali harmful queries
- `cross_bn_to_hi` → Use Bengali vectors on Hindi generation
  - Example: Bengali refusal vectors applied to Hindi harmful queries

### Total Combinations
```
3 models × 2 target_langs × 2 operations × 3 projections = 36 unique variants
```

---

## 🚀 Execution Steps

### Step 1: Identify Refusal Tokens (All 3 Models)

```bash
cd /Users/sravani/Documents/VSCode_projects/DS606_project_r2

# Run token identification for all 3 models + 2 languages
python scripts/identify_refusal_tokens_multimodel.py
```

**Output**:
- `config/refusal_tokens_multimodel.json` with structure:
  ```json
  {
    "tokens": {
      "llama3.1-8b": {
        "hi": [84310, 87244, ...],
        "bn": [72258, 146026, ...]
      },
      "qwen2.5-7b": { ... },
      "qwen2-7b": { ... }
    },
    ...
  }
  ```

**Expected runtime**: ~15-30 minutes (depends on GPU)

---

### Step 2: Generate All 36 Variants

```bash
# Generate with default options (all models, langs, operations, projections)
python scripts/generate_with_refusal_vectors_extended.py --overwrite

# Or custom subset:
python scripts/generate_with_refusal_vectors_extended.py \
  --models llama3.1-8b,qwen2.5-7b \
  --target-langs hi,bn \
  --operations add,subtract \
  --projections same_lang,cross_hi_to_bn,cross_bn_to_hi \
  --batch-size 16 \
  --max-new-tokens 256 \
  --overwrite
```

**Output Structure**:
```
output/refusal_vectors/
├── llama3.1-8b/
│   ├── hi/
│   │   ├── add/
│   │   │   ├── same_lang/
│   │   │   │   └── harmful_harm_ablation_evaluations.json (1000 completions)
│   │   │   ├── cross_bn_to_hi/
│   │   │   │   └── harmful_harm_ablation_evaluations.json
│   │   │   └── cross_hi_to_bn/  (N/A for hi target - empty)
│   │   └── subtract/
│   │       ├── same_lang/
│   │       ├── cross_bn_to_hi/
│   │       └── cross_hi_to_bn/
│   └── bn/
│       ├── add/
│       │   ├── same_lang/
│       │   ├── cross_hi_to_bn/
│       │   └── cross_bn_to_hi/  (N/A for bn target - empty)
│       └── subtract/
│           ├── same_lang/
│           ├── cross_hi_to_bn/
│           └── cross_bn_to_hi/
├── qwen2.5-7b/ (same structure)
└── gemma2-9b/ (same structure)
```

**Expected runtime**: ~8-12 hours for all 36 variants on 1× RTX 6000 GPU

---

### Step 3: Evaluate All Variants

```bash
# Evaluate all variants with WildGuard and Gemma-3
python scripts/evaluate_refusal_vectors.py

# Custom subset:
python scripts/evaluate_refusal_vectors.py \
  --completions-dir output/refusal_vectors \
  --output-dir output/refusal_vectors_evaluation \
  --models llama3.1-8b,qwen2.5-7b \
  --target-langs hi,bn
```

**Output**:
- `output/refusal_vectors_evaluation/`
  - `refusal_vectors_results.json` → Full results
  - `refusal_vectors_comparison.csv` → All metrics in one table
  - `wildguard_asr_pivot.csv` → WildGuard ASR pivot by model/lang/operation/projection
  - `gemma3_asr_pivot.csv` → Gemma-3 ASR pivot by model/lang/operation/projection

**Expected runtime**: ~4-6 hours (depends on evaluator speed)

---

## 📈 Understanding Results

### WildGuard ASR Pivot Example
```
                                              add                     subtract
                          same_lang cross_hi_to_bn same_lang cross_hi_to_bn ...
Model        Target Lang
llama3.1-8b  HI             15.2       18.5         22.1       20.3
             BN             14.8       85.2         21.5       88.1
qwen2.5-7b   HI             12.3       16.9         19.7       22.1
             BN             13.1       89.4         20.2       91.3
...
```

### Interpretation
- **Cross-lingual transfer success** → High ASR on cross-lingual (e.g., cross_hi_to_bn)
- **Operation comparison** → Subtract should have higher ASR than add (removes refusal)
- **Language effect** → Compare hi vs bn to see language-specific behaviors

---

## 🔍 Key Examples

### Example 1: Same-Language Hindi Refusal
```python
# Llama-3.1-8B, Hindi target, Add operation, Same language
# Uses: Hindi refusal tokens → applied to Hindi generation
# Path: output/refusal_vectors/llama3.1-8b/hi/add/same_lang/
# Expected: ~15-20% ASR (strong refusal, low success rate)
```

### Example 2: Cross-Lingual Transfer (Hindi → Bengali)
```python
# Llama-3.1-8B, Bengali target, Add operation, Hindi→Bengali
# Uses: Hindi refusal tokens → applied to Bengali generation
# Path: output/refusal_vectors/llama3.1-8b/bn/add/cross_hi_to_bn/
# Expected: May be higher ASR (~70-90%) if transfer fails
```

### Example 3: Vector Subtraction (Weaken Refusal)
```python
# Qwen2.5-7B, Hindi target, Subtract operation, Same language
# Uses: Hindi refusal tokens → subtracted from activations
# Path: output/refusal_vectors/qwen2.5-7b/hi/subtract/same_lang/
# Expected: Higher ASR (~60-80%) compared to add (~15-20%)
```

---

## 📋 Comparison Workflow

### Baseline (Already done)
```bash
# Run baseline generation (no refusal vectors)
python scripts/generate_baseline_vllm.py --output-dir output/asr_baseline_vllm

# Evaluate baseline
python scripts/evaluate_hi_bn_en_asr.py --output-dir output/asr_baseline_results
```

### Post-Refusal (This extension)
```bash
# Generate all variants (36 combinations)
python scripts/generate_with_refusal_vectors_extended.py

# Evaluate all variants
python scripts/evaluate_refusal_vectors.py
```

### Comparison Analysis
```python
# Compare baseline vs post-refusal
baseline_csv = "output/asr_baseline_results/asr_summary.csv"  # 9 rows (3 models × 3 langs)
refusal_csv = "output/refusal_vectors_evaluation/refusal_vectors_comparison.csv"  # 216 rows (36 combos × 2 evaluators × 3 models)

# Key metrics to compare:
# 1. Baseline ASR vs Add (should be lower)
# 2. Baseline ASR vs Subtract (should be higher)
# 3. Cross-lingual transfer rates (hi→bn vs same_lang)
```

---

## 🎯 Expected Findings

Based on refusal vector literature:

1. **Operation effect**:
   - Add ↓ ASR (strengthen refusal) → ~10-20%
   - Subtract ↑ ASR (weaken refusal) → ~50-80%

2. **Cross-lingual transfer**:
   - Successful: Hindi vectors transfer to Bengali (~70%+ reduction in ASR)
   - Limited by tokenization, embedding space overlap

3. **Model differences**:
   - Smaller models (Qwen2.5) may have stronger monolingual refusals
   - Larger models (Llama3.1) may have better cross-lingual alignment

---

## 📝 Notes

- All 36 variants are independent; can run in parallel on multiple GPUs
- Token IDs are model-specific (don't mix Llama refusal tokens with Qwen models)
- Cross-lingual projections test multilingual model capabilities
- Evaluation uses zero-shot classifiers (no fine-tuning)

---

## 🚨 Troubleshooting

**Error: "No refusal tokens for model/lang"?**
→ Run Step 1 first to identify tokens

**Error: "Completion file not found"?**
→ Run Step 2 to generate variants

**Memory issues?**
→ Reduce batch_size or run fewer models at once

**Slow evaluation?**
→ Evaluators are GPU-bound; expected with 36 variants

---

## Next Steps

1. ✅ Fixed model wrappers for all 8 models
2. ✅ Added subtraction hook
3. ✅ Extended tokens to 3 models (Llama3.1, Qwen2.5, Qwen2)
4. ✅ Created cross-lingual projection configs
5. ✅ Built comprehensive generation and evaluation scripts
6. 👉 **Now**: Execute the three steps above to generate results

Good luck! 🚀
