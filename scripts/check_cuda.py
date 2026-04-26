#!/usr/bin/env python3
"""
Diagnostic script to check CUDA compatibility with PyTorch
"""

import torch
import sys

print("=" * 80)
print("CUDA COMPATIBILITY DIAGNOSTIC")
print("=" * 80)

# Check CUDA availability
print("\n1. CUDA Availability:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"\n2. GPU Information:")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"      Capability: {torch.cuda.get_device_capability(i)}")
    
    print(f"\n3. Current GPU:")
    print(f"   Device: {torch.cuda.current_device()}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
else:
    print("\n   ❌ CUDA is not available!")

print("\n4. Issue Analysis:")
if torch.cuda.is_available() and not torch._C._cuda_isDeviceCapabilitySupported(torch.cuda.get_device_capability()):
    print("   ❌ CUDA kernel compatibility issue detected")
    print("   This typically means PyTorch was compiled for a different GPU architecture")
    print("\n   Solutions:")
    print("   1. Run in CPU mode (slower but compatible):")
    print("      CUDA_VISIBLE_DEVICES='' python scripts/identify_refusal_tokens.py")
    print("\n   2. Reinstall PyTorch for your CUDA version:")
    print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu11x")
    print("      (Replace cu11x with your CUDA version: cu118, cu121, etc.)")
else:
    print("   ✅ CUDA appears compatible")

print("\n" + "=" * 80)
