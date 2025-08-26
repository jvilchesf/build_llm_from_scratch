from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', env_file_encoding='utf-8'
    )

    # Dataset
    batch_size: int  # num of batch the all text will be splitted
    context_lenght: int
    num_workers: int
    stride: int
    drop_last: bool
    shuffle: bool

    # Embeddings
    emb_dim : int  # num embeddings represent number of dimensios will represent a token
    vocab_size: int
    output_dim: int

    # mha
    qkv_bias: bool  # linear layer bias activation yes or not for mha mechanism
    dropout_rate: float  # define % of random values to drop
    num_heads: int  # Num heads mha
    num_transformer_blocks: int
