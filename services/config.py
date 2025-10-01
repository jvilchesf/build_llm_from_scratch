from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ENV_PATH = Path(__file__).with_name("settings.env")  # services/settings.env


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8")

    # Dataset
    batch_size: int = 8
    context_length: int = 128
    num_workers: int = 0
    stride: int = 4
    drop_last: bool = True
    shuffle: bool = True

    # Embeddings
    emb_dim: int = 720
    vocab_size: int = 50257
    output_dim: int = 720

    # MHA
    qkv_bias: bool = True
    drop_rate: float = 0.1
    n_heads: int = 12
    n_layers: int = 12

    # training
    num_epochs: int = 100
    num_batch_evaluate: int = 20
    freq_eva: int = 5
