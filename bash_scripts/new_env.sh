conda create -n ds606_refusal python=3.11 -y
conda activate ds606_refusal

pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install "transformers==4.44.2" "accelerate==0.31.0" "huggingface-hub==0.23.3" \
            "datasets==2.19.2" "pandas==2.2.2" "tqdm==4.66.4" "python-dotenv==1.0.1" \
            "sentencepiece==0.2.0" "safetensors==0.4.3"