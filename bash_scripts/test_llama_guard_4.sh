#!/bin/bash
# Quick test to verify Llama Guard 4 setup

echo "Testing Llama Guard 4 Environment"
echo "=================================="

cd /users/student/prjstu/sravani.gunnu/DS606_project_r2

# Activate conda
source /users/student/prjstu/sravani.gunnu/miniconda3/etc/profile.d/conda.sh
conda activate ds606

export PYTHONPATH=/users/student/prjstu/sravani.gunnu/DS606_project_r2:$PYTHONPATH

echo ""
echo "1. Checking PyTorch..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"

echo ""
echo "2. Checking Transformers..."
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; print('✓ Transformers available')"

echo ""
echo "3. Testing model load (this may take 1-2 min)..."
python << 'EOF'
import sys
try:
    print("Loading processor...")
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("meta-llama/Llama-Guard-4-12B")
    print("✓ Processor loaded")
    
    print("Loading model...")
    from transformers import AutoModelForCausalLM
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-Guard-4-12B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✓ Model loaded")
    
    # Test classification
    print("\nTesting classification...")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "I'm doing well, thank you for asking!"}]},
    ]
    
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    response = proc.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
    print(f"✓ Classification test passed. Model says: {response}")
    
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
echo "Setup verification complete!"
