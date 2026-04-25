# Implementation Guide: Hindi and Bengali Language Support

This guide walks you through implementing the "Refusal Direction is Universal Across Safety-Aligned Languages" paper for Hindi (hi) and Bengali (bn) languages.

## What Has Been Done

### 1. Data Setup ✅
- Copied Hindi dataset files from `PolyRefuse/hi/` to `dataset/splits_multi/`:
  - `harmful_train_translated_hi.json`
  - `harmful_val_translated_hi.json`
  - `harmful_test_translated_hi.json`
  - `harmless_train_translated_hi.json`
  - `harmless_val_translated_hi.json`
  - `harmless_test_translated_hi.json`

- Copied Bengali dataset files from `PolyRefuse/bn/` to `dataset/splits_multi/`:
  - `harmful_train_translated_bn.json`
  - `harmful_val_translated_bn.json`
  - `harmful_test_translated_bn.json`
  - `harmless_train_translated_bn.json`
  - `harmless_val_translated_bn.json`
  - `harmless_test_translated_bn.json`

### 2. Configuration Files Created ✅
- Created `pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml` with:
  - `source_lang: hi` (Hindi language code)
  - `lang: hi` (Language parameter for tokenizer)
  - All other settings from the Japanese template

- Created `pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml` with:
  - `source_lang: bn` (Bengali language code)
  - `lang: bn` (Language parameter for tokenizer)
  - All other settings from the Japanese template

## What Needs to Be Done

### Step 1: Identify Refusal Tokens for Hindi and Bengali

Refusal tokens are language-specific token IDs that represent the first tokens a model generates when refusing a harmful request. They're crucial for the direction extraction process.

**Option A: Run the Automated Refusal Token Identification Script (Recommended)**

```bash
python scripts/identify_refusal_tokens.py
```

This script will:
1. Load the Qwen2.5-7B-Instruct model
2. Generate responses for harmful prompts in Hindi and Bengali
3. Extract the first tokens from these responses
4. Show you which token IDs are most common
5. Provide recommendations for refusal tokens

**Option B: Manual Identification**

If you want to manually identify tokens:

1. Run the model with a harmful prompt in Hindi/Bengali
2. Look at the first token it generates (before refusing)
3. Get the token ID from the tokenizer

Example in Python:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# For Hindi refusals
token_str = "मुझे"  # Common Hindi refusal start
token_id = tokenizer.encode(token_str)[0]
print(f"Token ID for '{token_str}': {token_id}")

# For Bengali refusals
token_str = "আমি"  # Common Bengali refusal start
token_id = tokenizer.encode(token_str)[0]
print(f"Token ID for '{token_str}': {token_id}")
```

### Step 2: Add Refusal Tokens to Model Implementations

Once you have the refusal token IDs from Step 1, update all model implementation files that support language-specific refusal tokens:

#### For Qwen2.5 Model (`pipeline/model_utils/qwen2_model.py`)

Find the `REFUSAL_TOKENS_LANG` dictionary and add Hindi and Bengali:

```python
REFUSAL_TOKENS_LANG = {
    'en': [40, 2121],  # ['I', 'As']
    'zh': [35946],     # '我'
    'de': [17360, 40369],  # 'Es'
    'th': [126331, 124430],  # 'ขอ', 'ฉ'
    'yi': [129613],
    'yo': [25612],
    'ja': [127748, 128976],
    'ru': [85391, 30174],
    'ko': [132759],
    'hi': [<TOKEN_ID_1>, <TOKEN_ID_2>],  # Replace with actual Hindi tokens
    'bn': [<TOKEN_ID_1>, <TOKEN_ID_2>],  # Replace with actual Bengali tokens
}
```

#### For Llama3 Model (`pipeline/model_utils/llama3_model.py`)

Similar update to the `REFUSAL_TOKENS_LANG` dictionary:

```python
REFUSAL_TOKENS_LANG = {
    'en': [40],        # ['I']
    'zh': [37046],     # '我'
    'de': [41469],     # 'Ich'
    'th': [104365],    # 'ฉ'
    'yi': [59610],
    'yo': [40],
    'ru': [86491],
    'ko': [101464],
    'ja': [122571],
    'hi': [<TOKEN_ID_1>],  # Replace with actual Hindi token
    'bn': [<TOKEN_ID_1>],  # Replace with actual Bengali token
}
```

#### For Other Models

Apply similar updates to:
- `pipeline/model_utils/qwen_model.py`
- `pipeline/model_utils/gemma_model.py`
- `pipeline/model_utils/gemma2_model.py`
- `pipeline/model_utils/yi_model.py`
- `pipeline/model_utils/yi1_5_model.py`

**Note:** Some models like Llama-2 might not support multilingual tokenization. Check if `lang` parameter is used in `_get_refusal_toks()` method.

### Step 3: Run the Pipeline

Once refusal tokens are configured, run the pipeline for Hindi and Bengali:

#### For Hindi:
```bash
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml
```

#### For Bengali:
```bash
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/bn/bn.yaml
```

**What the Pipeline Does:**

1. **Load and Filter Data**: Loads Hindi/Bengali harmful and harmless prompts, filters them based on refusal scores
2. **Generate Directions**: Computes difference-in-means vectors between harmful and harmless embedding activations
3. **Select Direction**: Selects the most effective refusal direction using validation set
4. **Ablate Refusal**: Removes the refusal direction from the model's activations
5. **Evaluate**: Tests on jailbreak and evaluation benchmarks
6. **Generate Completions**: Creates model responses for evaluation

### Step 4: Cross-Lingual Evaluation (Optional)

To test cross-lingual transferability (i.e., whether Hindi refusal directions work in Bengali and vice versa):

Create additional config files:
- `pipeline/runs/Qwen2.5-7B-Instruct/hi/hi_test_bn.yaml` - Use Hindi direction on Bengali data
- `pipeline/runs/Qwen2.5-7B-Instruct/bn/bn_test_hi.yaml` - Use Bengali direction on Hindi data

Update the configs to point to different language test sets and use pre-extracted directions from the other language.

## Pipeline Architecture Overview

### Key Components

1. **Dataset Loading** (`dataset/load_dataset.py`)
   - Loads prompts from multilingual data
   - Supports language parameter via filename pattern

2. **Model Factory** (`pipeline/model_utils/model_factory.py`)
   - Creates appropriate model wrapper based on model type
   - Passes language code to model

3. **Model Implementations** (`pipeline/model_utils/`)
   - Each model type (Qwen2, Llama3, Gemma2, etc.) has its own wrapper
   - Handles:
     - Model loading and tokenization
     - Language-specific refusal tokens
     - Chat template formatting
     - Hook installation for intervention

4. **Direction Generation** (`pipeline/submodules/generate_directions.py`)
   - Computes difference-in-means vectors:
     - `r_{l,i} = v^{harmful}_{l,i} - v^{harmless}_{l,i}`
   - Runs across all layers and token positions

5. **Direction Selection** (`pipeline/submodules/select_direction.py`)
   - Selects top N directions based on effectiveness
   - Evaluates candidates by measuring refusal score reduction
   - Applies KL divergence constraints to avoid affecting other capabilities

6. **Evaluation** (`pipeline/submodules/evaluate_jailbreak.py`)
   - Uses multiple methodologies:
     - Substring matching (checks for refusal keywords)
     - WildGuard (classifier-based evaluation)

### Main Pipeline Flow

```
run_pipeline.py
├── Load config
├── Initialize model with language
├── Load and sample datasets (with language parameter)
├── Filter data based on refusal scores
├── Generate candidate directions
├── Select best direction
├── Apply ablation/activation addition hooks
├── Generate completions on test sets
├── Evaluate with multiple methodologies
└── Save results
```

## Configuration File Parameters

Key parameters in `hi.yaml` and `bn.yaml`:

| Parameter | Meaning |
|-----------|---------|
| `source_lang` | Language code for loading data (hi/bn) |
| `lang` | Language code for tokenizer/refusal tokens (hi/bn) |
| `model_path` | Hugging Face model ID |
| `n_train` | Number of training samples per category (default: 128) |
| `n_val` | Number of validation samples (default: 32) |
| `n_test` | Number of test samples (default: 128) |
| `batch_size` | Training batch size (default: 64) |
| `ablation_coeff` | Strength of direction removal (default: 1.0) |
| `addact_coeff` | Strength of direction addition (default: 1.0) |
| `filter_train` | Whether to filter training data by refusal score |
| `filter_val` | Whether to filter validation data by refusal score |
| `jailbreak_evaluation_datasets` | Datasets to evaluate on after ablation |
| `artifact_path` | Where to save results |

## Troubleshooting

### Issue: KeyError for Hindi/Bengali refusal tokens
**Solution:** Make sure you've updated all model files' `REFUSAL_TOKENS_LANG` dictionaries with Hindi ('hi') and Bengali ('bn') entries.

### Issue: Dataset not found errors
**Solution:** Verify that Hindi and Bengali JSON files are in `dataset/splits_multi/` with correct naming:
- Should end with `_translated_hi.json` and `_translated_bn.json`

### Issue: Model runs but produces no ablation effect
**Possible causes:**
1. Incorrect refusal tokens (extraction picks completely wrong tokens)
2. Token filtering removing important examples
3. KL divergence threshold too tight

**Solution:** 
- Verify refusal tokens with `scripts/identify_refusal_tokens.py`
- Check `filter_train` and `filter_val` settings
- Adjust `kl_threshold` parameter

### Issue: Out of memory
**Solution:** Reduce `batch_size` in config file (try 32 or 16 instead of 64)

## Output Structure

After running the pipeline, results are saved to:
```
output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct/hi/TIMESTAMP/1/
├── config_run.yaml                    # Configuration used
├── direction_metadata_ablation.json   # Layer and position of direction
├── direction_ablation.pt              # The refusal direction vector
├── completions/
│   ├── harmful_harm_ablation_completions.json
│   ├── harmful_harm_ablation_evaluations.json
│   ├── jailbreakbench_harm_ablation_completions.json
│   ├── jailbreakbench_harm_ablation_evaluations.json
│   └── ...
├── generate_directions/
│   └── mean_diffs.pt                  # All candidate directions
├── select_direction/
│   └── ...
└── output.log                         # Execution log
```

## Expected Results

When the refusal direction works correctly, you should see:

1. **Baseline compliance rate**: % of harmful requests model refuses
2. **Post-ablation compliance rate**: Should be significantly higher (jailbreak attempts succeed)
3. **Cross-language transfer**: If using directions from one language on another

Example from paper (Table 1):
- English model: ~25-30% baseline compliance on harmful data
- After English direction ablation: 75-90% compliance (more jailbreaks succeed)
- When testing cross-linguistically, similar patterns emerge across languages

## References

- Paper: "Refusal Direction is Universal Across Safety-Aligned Languages"
- Main pipeline: `pipeline/run_pipeline.py`
- Model implementations: `pipeline/model_utils/`
- Paper methodology sections: 3 (Background), 5.1-5.4 (Main experiments)

## Next Steps

1. Run `scripts/identify_refusal_tokens.py` to get Hindi/Bengali token IDs
2. Update model files with refusal tokens
3. Run pipeline for each language
4. Analyze results in output directories
5. (Optional) Test cross-lingual transfer
