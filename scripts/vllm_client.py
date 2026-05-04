"""
Modular vLLM client for efficient batch generation with proper cache management.

Usage:
    from vllm_client import VLLMClient
    
    client = VLLMClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        num_gpus=1,
        download_dir="~/.cache/huggingface/transformers"
    )
    
    responses = client.generate_responses(
        prompts=["Prompt 1", "Prompt 2"],
        temperature=1.0,
        max_tokens=256,
        top_p=1.0
    )
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any


def setup_huggingface_cache():
    """Ensure writable Hugging Face cache directories."""
    hf_default = os.path.expanduser("~/.cache/huggingface")
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = hf_default
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_default, "transformers")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)


def setup_offline_mode(offline: bool = False):
    """Configure offline/cache-only mode if requested."""
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")


class VLLMClient:
    """
    Modular client for vLLM-based generation with batching support.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        num_gpus: Number of GPUs for tensor parallelism (default: 1)
        download_dir: Cache directory for model weights (default: ~/.cache/huggingface/transformers)
        max_model_len: Maximum model length (default: 4096)
        gpu_memory_util: GPU memory utilization fraction (default: 0.90)
        offline: Use offline/cache-only mode (default: False)
    """
    
    def __init__(
        self,
        model_name: str,
        num_gpus: int = 1,
        download_dir: Optional[str] = None,
        max_model_len: int = 4096,
        gpu_memory_util: float = 0.90,
        offline: bool = False,
    ):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.max_model_len = max_model_len
        self.gpu_memory_util = gpu_memory_util
        
        # Setup cache
        setup_huggingface_cache()
        setup_offline_mode(offline)
        
        if download_dir is None:
            download_dir = os.path.join(os.environ["TRANSFORMERS_CACHE"])
        self.download_dir = os.path.expanduser(download_dir)
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Suppress transformers warnings
        self._suppress_warnings()
        
        # Initialize vLLM
        self.llm = self._initialize_llm()
    
    def _suppress_warnings(self):
        """Suppress transformers & vLLM verbose output."""
        try:
            from transformers import logging
            logging.set_verbosity_error()
        except ImportError:
            pass
    
    def _initialize_llm(self):
        """Initialize vLLM engine with proper error handling."""
        try:
            from vllm import LLM
        except ImportError:
            raise RuntimeError(
                "vLLM not installed. Install with: "
                "pip install vllm>=0.5.0"
            )
        
        print(f"[vLLM] Initializing model: {self.model_name}")
        print(f"[vLLM]   Download dir: {self.download_dir}")
        print(f"[vLLM]   Tensor parallel size: {self.num_gpus}")
        print(f"[vLLM]   Max model length: {self.max_model_len}")
        
        try:
            llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.num_gpus,
                download_dir=self.download_dir,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_util,
                dtype="bfloat16",
            )
            print(f"[vLLM] ✓ Model loaded successfully")
            return llm
        except Exception as e:
            print(f"[vLLM] ✗ Failed to initialize model: {e}")
            raise
    
    def generate_responses(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        use_tqdm: bool = True,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            temperature: Sampling temperature (0.0 = deterministic, 1.0+ = more random)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate per response
            use_tqdm: Show progress bar
        
        Returns:
            List of generated response strings
        """
        try:
            from vllm import SamplingParams
        except ImportError:
            raise RuntimeError("vLLM not installed")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        print(f"[vLLM] Generating {len(prompts)} completions...")
        print(f"[vLLM]   Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}")
        
        try:
            outputs = self.llm.generate(
                prompts,
                sampling_params,
                use_tqdm=use_tqdm,
            )
            responses = [
                output.outputs[0].text.strip() if output.outputs else ""
                for output in outputs
            ]
            print(f"[vLLM] ✓ Generated {len(responses)} completions")
            return responses
        except Exception as e:
            print(f"[vLLM] ✗ Generation failed: {e}")
            raise
    
    def __del__(self):
        """Cleanup vLLM resources."""
        if hasattr(self, "llm"):
            try:
                del self.llm
            except Exception:
                pass


def main():
    """Quick test of VLLMClient."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vLLM client")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=2, help="Number of test prompts")
    parser.add_argument("--offline", action="store_true", help="Use offline mode")
    args = parser.parse_args()
    
    try:
        client = VLLMClient(
            model_name=args.model,
            num_gpus=args.num_gpus,
            offline=args.offline,
        )
        
        test_prompts = [
            f"Translate 'hello' to language {i}" for i in range(args.prompts)
        ]
        
        responses = client.generate_responses(
            prompts=test_prompts,
            temperature=0.7,
            max_tokens=50,
        )
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
            print(f"\n[{i}] PROMPT:\n{prompt}\n\n[{i}] RESPONSE:\n{response}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
