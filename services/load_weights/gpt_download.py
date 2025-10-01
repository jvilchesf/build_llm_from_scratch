import os
import urllib.request
from pathlib import Path


def download_file(file_name: str):
    # Download File

    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

    # Get current script directory
    SCRIPT_DIR = Path(__file__).parent

    # Target path in the same folder as the script
    target_path = SCRIPT_DIR / file_name

    if not target_path.exists():
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded to {file_name}")
