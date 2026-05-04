# Quick Reference: Refusal Vectors Extension

## 🚀 Quick Commands

```bash
# Step 1: Identify Tokens (15-30 min)
python scripts/identify_refusal_tokens_multimodel.py

# Step 2: Generate All 36 Variants (8-12 hours)
python scripts/generate_with_refusal_vectors_extended.py --overwrite

# Step 3: Evaluate Everything (4-6 hours)
python scripts/evaluate_refusal_vectors.py
```

## 📊 View Results

```bash
# CSV comparison table (all metrics in one place)
cat output/refusal_vectors_evaluation/refusal_vectors_comparison.csv

# WildGuard pivot table (by model/lang/operation/projection)
cat output/refusal_vectors_evaluation/wildguard_asr_pivot.csv

# Gemma-3 pivot table
cat output/refusal_vectors_evaluation/gemma3_asr_pivot.csv

# Full JSON results
cat output/refusal_vectors_evaluation/refusal_vectors_results.json
```

## 🎯 Custom Runs

```bash
# Only Llama models
python scripts/generate_with_refusal_vectors_extended.py \
  --models llama3.1-8b,llama2-model \
  --overwrite

# Only addition (add) operation
python scripts/generate_with_refusal_vectors_extended.py \
  --operations add \
  --overwrite

# Only same-language projections
python scripts/generate_with_refusal_vectors_extended.py \
  --projections same_lang \
  --overwrite

# Only Hindi target language
python scripts/generate_with_refusal_vectors_extended.py \
  --target-langs hi \
  --overwrite
```

## 📈 Expected Outputs

After **Step 3**, you'll have:
```
output/refusal_vectors_evaluation/
├── refusal_vectors_results.json            ← Full JSON results
├── refusal_vectors_comparison.csv          ← 216 rows (all variants evaluated)
├── wildguard_asr_pivot.csv                 ← WildGuard ASR by model/lang/op
└── gemma3_asr_pivot.csv                    ← Gemma-3 ASR by model/lang/op
```

## 🔍 Key Metrics to Compare

| Metric | Meaning | Expected Range |
|--------|---------|-----------------|
| add/same_lang ASR | Baseline with refusal vector added (strengthened) | 10-20% |
| subtract/same_lang ASR | With refusal vector subtracted (weakened) | 60-80% |
| cross_hi_to_bn ASR | Hindi vectors on Bengali (transfer effect) | Variable |
| cross_bn_to_hi ASR | Bengali vectors on Hindi (transfer effect) | Variable |

## 📋 Variant Legend

```
Example: llama3.1-8b/hi/add/same_lang
         ├─ Model: llama3.1-8b
         ├─ Target Language: hi (Hindi)
         ├─ Operation: add (strengthen refusal)
         └─ Projection: same_lang (Hindi vectors on Hindi)

Example: qwen2.5-7b/bn/subtract/cross_hi_to_bn
         ├─ Model: qwen2.5-7b
         ├─ Target Language: bn (Bengali)
         ├─ Operation: subtract (weaken refusal)
         └─ Projection: cross_hi_to_bn (Hindi vectors on Bengali)
```

## 🎯 Analysis Workflow

```python
# 1. Load comparison CSV
import pandas as pd
df = pd.read_csv('output/refusal_vectors_evaluation/refusal_vectors_comparison.csv')

# 2. Filter by model
llama_results = df[df['Model'] == 'llama3.1-8b']

# 3. Compare operations
add_results = llama_results[llama_results['Operation'] == 'add']
subtract_results = llama_results[llama_results['Operation'] == 'subtract']

# 4. Cross-lingual transfer
cross_transfer = llama_results[llama_results['Projection'].str.contains('cross')]

# 5. Calculate improvements
baseline_asr = 85.0  # From baseline evaluation
add_improvement = baseline_asr - add_results['WildGuard ASR (%)'].mean()
subtract_degradation = subtract_results['WildGuard ASR (%)'].mean() - baseline_asr
```

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: refusal_tokens_multimodel.json` | Run Step 1 first |
| `No completion file found` | Run Step 2 for that variant |
| `CUDA out of memory` | Reduce `--batch-size` (default: 8, try 4) |
| `Model not found` | Ensure models are cached or internet available |
| `Evaluator timeout` | Evaluators are slow; this is normal (4-6 hours) |

## 💡 Tips

- All 36 variants are independent → Can run Step 2 on multiple GPUs
- Each variant takes ~12-15 min to generate on 1× RTX 6000
- Evaluation is sequential (no parallelization) → Expect 4-6 hours
- Token identification is one-time → Reuse results across runs
- For smaller experiments: Use `--sample-size 100` (default: all ~1000)

## 📚 Full Documentation

- **Detailed guide**: `REFUSAL_VECTORS_EXTENSION_GUIDE.md`
- **Implementation summary**: `IMPLEMENTATION_SUMMARY.md`
- **Original refusal flow**: `REFUSAL_TOKEN_FLOW.md`

---

**Ready?** Run Step 1, then Step 2, then Step 3! 🚀
