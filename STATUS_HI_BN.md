# Implementation Status: Hindi & Bengali Refusal Direction

**Date:** April 26, 2026  
**Status:** ✅ Data and configuration setup complete. Ready for token identification.

---

## ✅ Completed Tasks

### 1. Data Integration
- ✅ Hindi translated prompts copied to `dataset/splits_multi/`:
  - `harmful_train_translated_hi.json` (68 KB)
  - `harmful_val_translated_hi.json` (13 KB)
  - `harmful_test_translated_hi.json` (333 KB)
  - `harmless_train_translated_hi.json` (45 KB)
  - `harmless_val_translated_hi.json` (44 KB)
  - `harmless_test_translated_hi.json` (160 KB)

- ✅ Bengali translated prompts copied to `dataset/splits_multi/`:
  - `harmful_train_translated_bn.json` (69 KB)
  - `harmful_val_translated_bn.json` (13 KB)
  - `harmful_test_translated_bn.json` (333 KB)
  - `harmless_train_translated_bn.json` (46 KB)
  - `harmless_val_translated_bn.json` (45 KB)
  - `harmless_test_translated_bn.json` (163 KB)

### 2. Configuration Files
- ✅ Created `pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml`
- ✅ Created `pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml`

### 3. Helper Scripts
- ✅ `scripts/identify_refusal_tokens.py` - Identifies language-specific refusal tokens
- ✅ `scripts/update_refusal_tokens.py` - Batch updates all model files with tokens

### 4. Documentation
- ✅ `QUICK_START_HI_BN.md` - Step-by-step quick start guide
- ✅ `IMPLEMENTATION_GUIDE_HI_BN.md` - Detailed technical guide

---

## 📋 Next Steps (In Order)

### Step 1: Identify Refusal Tokens (Required)

Run the token identification script:

```bash
cd /Users/sravani/Documents/VSCode_projects/DS606_project_r2
python scripts/identify_refusal_tokens.py
```

**Expected output:**
```
================================================================================
Identifying refusal tokens for Hindi (hi)
================================================================================
Loading model Qwen/Qwen2.5-7B-Instruct...
...
Recommended Hindi refusal tokens: [XXXXX, XXXXX]

================================================================================
Identifying refusal tokens for Bengali (bn)
================================================================================
Loading model Qwen/Qwen2.5-7B-Instruct...
...
Recommended Bengali refusal tokens: [XXXXX, XXXXX]
```

⏱️ **Time:** 5-10 minutes (requires GPU)

---

### Step 2: Update Model Files (Automated)

After getting the tokens, update all model files in one command:

```bash
# Example with hypothetical token IDs
python scripts/update_refusal_tokens.py --hi "127748,128976" --bn "12345,67890"
```

This will update:
- `pipeline/model_utils/qwen2_model.py`
- `pipeline/model_utils/llama3_model.py`
- `pipeline/model_utils/gemma2_model.py`
- `pipeline/model_utils/yi_model.py`
- `pipeline/model_utils/yi1_5_model.py`
- `pipeline/model_utils/gemma_model.py`

**Verification:**
```bash
git diff pipeline/model_utils/qwen2_model.py
```

Should show:
```python
REFUSAL_TOKENS_LANG = {
    ...existing...
    'hi': [127748, 128976],
    'bn': [12345, 67890],
}
```

⏱️ **Time:** < 1 minute

---

### Step 3: Run the Pipeline

#### For Hindi:
```bash
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml
```

#### For Bengali:
```bash
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml
```

**What happens:**
1. Loads Qwen2.5-7B-Instruct model
2. Processes harmful and harmless prompts in the target language
3. Extracts refusal vectors at each layer and token position
4. Identifies the best layer (where refusal is most localized)
5. Tests ablation on validation set
6. Generates results

**Outputs saved to:**
```
output/
└── Qwen/
    └── Qwen2.5-7B-Instruct/
        ├── hi/
        │   └── [timestamp]/
        │       ├── 1/
        │       │   ├── hi.yaml
        │       │   ├── position_analysis.json
        │       │   ├── refusal_vectors.pt
        │       │   ├── validation_results.json
        │       │   └── test_results.json
        │       └── ...
        └── bn/
            └── [similar structure]
```

⏱️ **Time:** 30-60 minutes per language (requires GPU)

---

### Step 4: Cross-Lingual Transfer Tests (Optional)

Test Hindi/Bengali refusal directions on other languages:

```bash
# Evaluate Hindi-extracted vectors on other languages
python -m scripts.multi_test --config output/Qwen/Qwen2.5-7B-Instruct/hi/[timestamp]/1/hi.yaml
```

This will show:
- How well Hindi refusal vectors work in other languages
- Cross-lingual transfer effectiveness
- Universality of the refusal direction

⏱️ **Time:** 10-20 minutes per target language

---

## 📊 Expected Results

Based on the paper's findings for other languages, you should see:

### Ablation Results (with refusal direction removed):
```
Language    Baseline  After Ablation  Increase
Hindi       ~10%      ~80-90%         +80%
Bengali     ~10%      ~80-90%         +80%
```

This means:
- Models with refusal direction removed become vulnerable to ~80-90% compliance rate
- Refusal direction is the primary mechanism preventing jailbreaks

### Cross-Lingual Transfer:
- Hindi vectors should be ~70-90% effective when ablated from Bengali outputs
- Bengali vectors should be ~70-90% effective when ablated from Hindi outputs
- This demonstrates the universality of refusal mechanisms

---

## 🔧 Troubleshooting

### Issue: "Model not found" error
```
OSError: Can't find model 'Qwen/Qwen2.5-7B-Instruct'
```
**Fix:** Login to HuggingFace
```bash
huggingface-cli login
# Enter your HuggingFace token when prompted
```

### Issue: Out of Memory (CUDA)
**Fix:** Modify `identify_refusal_tokens.py` line 36:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32,  # ← Change from float16
    device_map="cuda:0"
)
```

### Issue: Pipeline hangs or takes too long
**Details:** Pipeline is I/O bound during dataset loading. First run will be slowest.
**Solution:** Be patient or run with smaller dataset (edit config yaml to reduce samples)

---

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| `QUICK_START_HI_BN.md` | This quick reference |
| `IMPLEMENTATION_GUIDE_HI_BN.md` | Detailed technical explanations |
| `scripts/identify_refusal_tokens.py` | Token identification (RUN FIRST) |
| `scripts/update_refusal_tokens.py` | Batch token updates |
| `pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml` | Hindi pipeline config |
| `pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml` | Bengali pipeline config |

---

## 🎯 Success Criteria

✅ You'll know it's working when:

1. **Token identification script runs:** Outputs token IDs for Hindi and Bengali
2. **Model files update:** All 6 model files get the new tokens
3. **Pipeline starts:** No errors for 5+ minutes after running pipeline command
4. **Results appear:** `output/Qwen/Qwen2.5-7B-Instruct/hi/` and `bn/` directories fill with results
5. **Compliance rates change:** Ablation shows significant jailbreak increase (80%+)

---

## 📞 Support

For detailed explanations of:
- How refusal vectors work → See `IMPLEMENTATION_GUIDE_HI_BN.md` Section 3
- Configuration parameters → See `IMPLEMENTATION_GUIDE_HI_BN.md` Section 4
- Output format → See `IMPLEMENTATION_GUIDE_HI_BN.md` Section 5
- Visualization → See `IMPLEMENTATION_GUIDE_HI_BN.md` Section 6

---

**Ready to begin? Start with:** `python scripts/identify_refusal_tokens.py`
