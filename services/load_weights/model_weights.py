import torch
from pathlib import Path
import tiktoken

from services.model.gpt_model import GPT_backbone
from services.load_weights.gpt_download import download_file
from services.config import Settings


def get_model():
    """
    Update model configs based on the model downloaded
    """

    conf = Settings()
    tokenizer = tiktoken.get_encoding("gpt2")

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-xl (1558M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # ✅ update the conf instance with BASE_CONFIG
    conf = conf.model_copy(update=BASE_CONFIG)

    # file_name = "gpt2-small-124M.pth"
    # file_name = "gpt2-medium-355M.pth"
    # file_name = "gpt2-large-774M.pth"
    file_name = "gpt2-xl-1558M.pth"

    download_file(file_name)
    model = load_weights(conf, file_name)

    return model, conf, tokenizer


def load_weights(conf, file_name):
    # For llms_from_scratch installation instructions, see:
    # https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
    # Find the folder where this script lives
    script_dir = Path(__file__).parent

    # Construct full path to the weights file in the same folder
    weights_path = script_dir / file_name

    gpt = GPT_backbone(conf)
    gpt.load_state_dict(torch.load(weights_path, weights_only=True))
    gpt.eval()

    return gpt
