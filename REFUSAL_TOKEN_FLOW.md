# Refusal Token Flow: How They Work in the Pipeline

## Overview

Refusal tokens are language-specific tokens that the model generates first when refusing harmful requests. They are identified once and then used throughout the ablation pipeline.

---

## 1️⃣ **Token Identification Phase** (What you just did)

```bash
python scripts/identify_refusal_tokens.py
```

**Process:**
- Loads harmful prompts from the target language dataset
- Generates responses from the model
- Extracts the **first token** of each refusal response
- Identifies which tokens appear most frequently as refusal starters

**Output:**
- Hindi: `[198, 262, 87244, 145799, 151644]`
- Bengali: `[148014, 148204, 149525, 151644]`
- **Saved to:** `config/refusal_tokens_hi_bn.json`

---

## 2️⃣ **Configuration & Model Update Phase**

```bash
python scripts/update_refusal_tokens.py --hi "198,262,87244,145799,151644" --bn "148014,148204,149525,151644"
```

**What happens:**
- Updates all 6 model files with the tokens
- These are stored in `REFUSAL_TOKENS_LANG` dictionary in each model file

**Example in `qwen2_model.py`:**
```python
REFUSAL_TOKENS_LANG = {
    'en': [40, 2121],
    'hi': [198, 262, 87244, 145799, 151644],  # ← Added
    'bn': [148014, 148204, 149525, 151644],   # ← Added
    ...
}
```

---

## 3️⃣ **Pipeline Execution Phase**

```bash
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml
```

### **Inside the Pipeline (Step by Step):**

### **Step A: Data Preparation**
```python
# Load the refusal tokens for the target language
refusal_tokens = REFUSAL_TOKENS_LANG['hi']  # [198, 262, 87244, 145799, 151644]

# Load harmful and harmless prompts
harmful_prompts = load_from_file('dataset/splits_multi/harmful_train_translated_hi.json')
harmless_prompts = load_from_file('dataset/splits_multi/harmless_train_translated_hi.json')
```

### **Step B: Generate Responses**
```python
# For harmful prompts
harmful_response = model.generate(harmful_prompt)
# Example: harmful_response = [198, 87244, 165, 234, ...]
#                               ↑ First token is in refusal_tokens list!

# For harmless prompts  
harmless_response = model.generate(harmless_prompt)
# Example: harmless_response = [40, 165, 234, 562, ...]
#                                ↑ First token NOT in refusal_tokens list
```

### **Step C: Extract Refusal Vectors**
```
For each layer l and token position i:
    IF token_position is where refusal tokens appear:
        # Extract hidden activation from harmful response
        harmful_activation = model.hidden_states[layer_l][harmful_response]
        
        # Extract hidden activation from harmless response
        harmless_activation = model.hidden_states[layer_l][harmless_response]
        
        # Compute the difference vector (this is the "refusal direction")
        refusal_vector[l, i] = harmful_activation - harmless_activation
```

**Why refusal tokens matter here:**
- They tell the pipeline WHERE in the response the refusal signal is strongest
- Token position 0 (where refusal tokens appear) is the most important position
- The refusal vector is extracted precisely at these positions

### **Step D: Find Best Layer**
```python
# Calculate which layer has the strongest refusal signal
best_layer = argmax(magnitude(refusal_vector))
# Result: e.g., layer 12 has the clearest refusal direction

# Select the refusal vector from the best layer
best_refusal_vector = refusal_vector[best_layer]
```

### **Step E: Test Ablation (Remove Refusal)**
```python
# For validation prompts:
for prompt in validation_harmful_prompts:
    # Normal behavior
    response_normal = model.generate(prompt)
    
    # With refusal direction REMOVED (ablation)
    # Equation: x'_l = x_l - α * r̂_l
    response_ablated = model.generate_with_ablation(
        prompt, 
        ablation_layer=best_layer,
        ablation_vector=best_refusal_vector,
        alpha=1.0
    )
    
    # Check if response is now compliant (jailbroken)
    if is_compliant(response_ablated):
        jailbreak_count += 1

# Calculate compliance rate
compliance_rate = jailbreak_count / total_prompts
```

**Expected result:** Compliance jumps from ~10% to ~80-90% ✨

---

## Flow Diagram

```
Identify Tokens
      ↓
[198, 262, 87244, ...]
      ↓
Store in model files (REFUSAL_TOKENS_LANG)
      ↓
Pipeline loads tokens for target language
      ↓
Generate harmful/harmless responses
      ↓
Extract activations at token positions where refusal occurs
      ↓
Compute difference vectors (refusal directions)
      ↓
Test ablation (remove refusal direction)
      ↓
Measure compliance rate increase
      ↓
Results saved to output/
```

---

## File Tracking

| Stage | File | Purpose |
|-------|------|---------|
| **Identification** | `config/refusal_tokens_hi_bn.json` | Stores identified tokens (backup) |
| **Configuration** | `pipeline/model_utils/*_model.py` | Stores tokens in `REFUSAL_TOKENS_LANG` |
| **Pipeline** | Config YAML | References language code to load tokens |
| **Results** | `output/Qwen/.../hi/` | Saves extracted vectors and ablation results |

---

## Key Insight

Without knowing which tokens represent refusals, the pipeline would have to:
1. Analyze activations at ALL token positions (inefficient)
2. Not know which positions are semantically meaningful (poor signal)

**With refusal tokens**, the pipeline focuses on the exact positions where the refusal mechanism activates, making extraction precise and interpretable.

---

## Next Command

After running identification with 100 samples:

```bash
# Update model files with tokens
python scripts/update_refusal_tokens.py --hi "198,262,87244,145799,151644" --bn "148014,148204,149525,151644"

# Then run pipeline
python -m pipeline.run_pipeline --config pipeline/runs/Qwen2.5-7B-Instruct/hi/hi.yaml
```
