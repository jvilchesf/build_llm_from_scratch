import torch
import torch.nn as nn
from loguru import logger


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout,
        n_heads: int,
        qkv_bias: bool = False,
    ):
        """
        Class with logic to implement transformers multi head attention code
        Args:
            dim_in: Number of dimensions of the tokens embeddings
            dim_out: Number of dimensions of context output representations of imput
            context_length: Dimension of the attention weights matrix, it is where you weight all individual words in the sentences
                            because the idea is to mask out half of the matrix, it is neccesary to know the matrix dimmension.
            dropout: Porcentage of tokens to dropout from the attention weights matrix
            n_heads: Define number of heads, it referes number of times the multihead attention context will be apply
        """

        super().__init__()

        assert dim_out % n_heads == 0, (
            f"d_out ({dim_out}) must be divisible by n_heads ({n_heads}) "
        )

        self.dim_in = dim_in
        self.n_heads = n_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // n_heads
        # Declare query, key and value matrix's
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)  # [256 x 128]
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)  # [256 x 128]
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)  # [256 x 128]
        self.out_proj = nn.Linear(dim_out, dim_out)

        # Declare droput
        self.dropout = nn.Dropout(dropout)

        # Mask for causal attention
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, X):
        """
        Run forward step for each input "layer", thinking on the input X as a shelf where you receive post card, it is compose
        by different layers [3, 6, 5], this for example would be 3 layers (different batches) with each of them with a matrix of 6 x 5.

        Args:
        X: It is the input torch 3 dimensional matrix.
        """

        # getting dimensions and variables
        # logger.info(f"The input shape is: {X.shape}")
        batch_size, num_tokens, dim_out = X.shape

        # Initialize q, k, v for each batch
        query = self.W_query(
            X
        )  # X = [8 x 4 x 256] | W_query = [256 x 128] => 8 x4 x 128
        key = self.W_key(X)  # X = [8 x 4 x 256] | W_key = [256 x 128] => 8 x 4 x 128
        value = self.W_value(
            X
        )  # X = [8 x 4 x 256] | W_values = [256 x 128] => 8 x 4 x 128

        # Reshaping query and view matrices, including n_heads and head_dim
        query = query.view(
            batch_size, num_tokens, self.n_heads, self.head_dim
        )  # [8 x 4 x 8 x 16]
        key = key.view(
            batch_size, num_tokens, self.n_heads, self.head_dim
        )  # [8 x 4 x 8 x 16]
        value = value.view(
            batch_size, num_tokens, self.n_heads, self.head_dim
        )  # [8 x 8 x 4 x 16]

        # Reshape from (batch x num_tokens x n_heads x head_dim) to (batch_size x n_heads x num_tokens x head_dim)
        # What is here really happening is that the heads are changed by tokens and are now the main thing on each slice of the cube or matrix
        query = query.transpose(1, 2)  # [8 x 4 x 8 x 16] -> [8 x 8 x 4 x 16]
        key = key.transpose(1, 2)  # [8 x 4 x 8 x 16] -> [8 x 8 x 4 x 16]
        value = value.transpose(1, 2)  # [8 x 4 x 8 x 16] -> [8 x 8 x 4 x 16]

        # MUltiply query with keys
        # The real matrix multiplication is between heads and query-tokens. because multiplication is row by columns
        # it is necessary to change the second key matrix dimensions
        attention_scores = query @ key.transpose(
            2, 3
        )  # [8 x 8 x 4 x 16] @ [8 x 8 x 16 x 4] = [8 x 8 x 4 x 4]
        # Mask out weights to don't consider known words
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # attention_weights
        attention_weights = torch.softmax(
            attention_scores / key.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # Context Vector
        context_vector = (attention_weights @ value).transpose(
            1, 2
        )  # [8 x 8 x 4 x 4] @ [8 x 8 x 4 x 16] = [8 x 8 x 4 x 16].transpose(1, 2) -> [8 x 4 x 8 x 16]
        context_vector = context_vector.contiguous().view(
            batch_size, num_tokens, self.dim_out
        )  # [8 x 4 x 128] Reshaping to normal form to deliver output, combine heads, where self.dim_out = self.n_heads * self.head_dim
        context_vector = self.out_proj(
            context_vector
        )  # Adds an optional linear projection

        return context_vector
