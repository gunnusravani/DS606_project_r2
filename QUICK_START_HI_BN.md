# Quick Start Guide: Hindi & Bengali Implementation

This guide walks you through implementing the refusal direction analysis for Hindi and Bengali languages.

## Step 1: Identify Refusal Tokens

The first critical step is to identify which tokens the model uses when refusing harmful requests in Hindi and Bengali.

### Run Token Identification

```bash
cd /Users/sravani/Documents/VSCode_projects/DS606_project_r2
python scripts/identify_refusal_tokens.py
```

**What this does:**
- Loads the Qwen2.5-7B-Instruct model
- Generates responses to harmful prompts in Hindi and Bengali
- Identifies the first tokens in refusal responses
- Outputs token IDs like: `[12345, 67890]`

**Expected output format:**
```
================================================================================
Identifying refusal tokens for Hindi (hi)
================================================================================
Loading model Qwen/Qwen2.5-7B-Instruct...
Loading harmful prompts from .../harmful_train_translated_hi.json...
Loaded 5 harmful prompts
Generating responses...

Generated responses - First tokens:
Unique first token IDs: [127748, 128976]
  Token ID 127748: 'I'
  Token ID 128976: 'cannot'

Sample responses:
  Prompt 1: ...
  Response: I cannot help with that request...

Recommended Hindi refusal tokens: [127748, 128976]

================================================================================
Identifying refusal tokens for Bengali (bn)
================================================================================
...similar output...

Recommended Bengali refusal tokens: [xxxxx, yyyyy]
```

### Save the Token IDs
When the script completes, **write down the token IDs** it outputs:
- **Hindi tokens:** _____________________
- **Bengali tokens:** _____________________

---

## Step 2: Update Model Files with Tokens

Once you have the token IDs, add them to all model files:

### Files to Update:
1. `pipeline/model_utils/qwen2_model.py`
2. `pipeline/model_utils/llama3_model.py`
3. `pipeline/model_utils/gemma2_model.py`
4. `pipeline/model_utils/yi_model.py`
5. `pipeline/model_utils/yi1_5_model.py`
6. `pipeline/model_utils/gemma_model.py`

### Edit Each File

In each file, find the `REFUSAL_TOKENS_LANG` dictionary and add:

```python
REFUSAL_TOKENS_LANG = {
    # ...existing languages...
    'hi': [YOUR_HI_TOKENS_HERE],    # e.g., [127748, 128976]
    'bn': [YOUR_BN_TOKENS_HERE],    # e.g., [xxxxx, yyyyy]
}
```

---

## Step 3: Create Language-Specific Config Files

Configuration files are already created:
- ✅ `pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml`
- ✅ `pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml`

You can add more model configurations following the same pattern.

---

## Step 4: Run the Pipeline

### For Hindi:
```bash
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml
```

### For Bengali:
```bash
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml
```

**What the pipeline does:**
1. Extracts refusal direction vectors from the model
2. Identifies the best layer and token position for the refusal signal
3. Tests ablation (removing the refusal direction)
4. Generates a compliance rate showing how many harmful queries were answered

**Output location:**
```
output/
├── ja_vector_sweep/
├── Qwen/
│   └── Qwen2.5-7B-Instruct/
│       ├── hi/     ← Hindi results
│       ├── bn/     ← Bengali results
│       └── ...
```

---

## Step 5: Evaluate Across Languages (Optional)

To test Hindi/Bengali refusal vectors on other languages:

```bash
python -m scripts.multi_test --config output/ja_vector_sweep/Qwen/Qwen2.5-7B-Instruct/hi/20250519-232436/1/hi.yaml
```

This runs the Hindi-extracted refusal direction on models in other languages.

---

## Troubleshooting

### ❌ Model not found
```
OSError: Can't find model 'Qwen/Qwen2.5-7B-Instruct'
```
**Solution:** Check your HuggingFace token and internet connection
```bash
huggingface-cli login
```

### ❌ Out of memory (CUDA)
```python
# Modify identify_refusal_tokens.py
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32,  # Change to float32
    device_map="auto"
)
```

### ❌ File not found error
Ensure you're in the correct directory:
```bash
cd /Users/sravani/Documents/VSCode_projects/DS606_project_r2
```

---

## Expected Timeline

- **Token identification:** 5-10 minutes (GPU)
- **Pipeline execution:** 30-60 minutes per language
- **Multi-language eval:** 10-20 minutes per target language

---

## Next Steps After Completion

1. Collect results from `output/` directory
2. Generate visualizations (PCA plots, heatmaps)
3. Compare Hindi/Bengali performance with other languages
4. Analyze cross-lingual transferability

See `IMPLEMENTATION_GUIDE_HI_BN.md` for detailed technical information.
