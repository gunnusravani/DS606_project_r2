#!/bin/bash
# Install missing dependencies and fix CUDA issues

echo "=========================================="
echo "Installing Missing Dependencies"
echo "=========================================="

# Install lm-eval
echo ""
echo "📦 Installing lm-eval (evaluation harness)..."
pip install lm-eval

# Install other potentially missing packages
echo ""
echo "📦 Installing additional dependencies..."
pip install vllm --upgrade || echo "⚠️  vllm installation had issues (may not be needed for this pipeline)"

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "If you get CUDA errors, try:"
echo "  1. Run: CUDA_LAUNCH_BLOCKING=1 python -m pipeline.run_pipeline ..."
echo "  2. Or reinstall PyTorch for your CUDA version"
echo ""
