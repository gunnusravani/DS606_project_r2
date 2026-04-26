#!/usr/bin/env python3
"""
Quick test to verify GPU works with model generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Testing GPU generation...")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load a small model to test
model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded on GPU")

# Test generation
test_prompt = "Hello,"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

print(f"\nGenerating from: '{test_prompt}'")
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
print("\n✅ GPU generation test PASSED!")
