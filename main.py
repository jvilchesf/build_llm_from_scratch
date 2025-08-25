from loguru import logger
from create_data_loader import create_data_loader_v1
from multihead_attention import MultiHeadAttention
import os
import torch
import urllib.request


def main():
    """
    Main function to create train an LLM
    """
    # Download text to train
    filepath = 'source/the-veredict.txt'
    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
        urllib.request.urlretrieve(url, filepath)

    with open(filepath) as f:
        text = f.read()

    # Create dataset
    window_size = 4
    batch_size = 8
    dataloader = create_data_loader_v1(text,
                                       batch_size,
                                       window_size,
                                       drop_last=True,
                                       shuffle=True,
                                       num_workers=2,
                                       stride=4,
                                       )

    data_iter = iter(dataloader)
    input, target = next(data_iter)  # 8 x 4

    # Create weight embeddings
    # Embedding is a kind of high dimensional dictionary for each word
    vocab_size = 50257  # This value comes from the vocab size of tiktoken library
    output_dim = 256  # Variable represent the number of dimensions that each word will have in this new dynamic dictionary
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # Pasing input throught the embedding token_embedding_layer
    token_embeddings = token_embedding_layer(input)  # [4 x 8 x 256]

    # Posititonal embedding
    # Positional embedding is necessary to include information about place in the senteces
    # Sometimes you have repetated words in a sentence, with this pos embedding won't be assigned
    # same value to the same word
    pos_embedding_layer = torch.nn.Embedding(window_size, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(window_size))
    input_embedding = token_embeddings + pos_embeddings

    # Create Instance multihead_attention
    dim_in = input_embedding.shape[-1]  # Var created to define q, k and value matrices
    dim_out = 256 # Variable that represent the output dimensions of each q, k, v matrices 
    num_heads = 8 # Number of heads will define the division of dim_out in groups, with 128 and 4, I'll get 4 groups of 32 dimensiones on each q, k, v
    context_length = window_size

    # Initiallize Multhiead attention weights and variables
    mha = MultiHeadAttention(dim_in, dim_out, context_length, 0.0, num_heads)
    forward = mha.forward(input_embedding)

    print(forward.shape)

if __name__ == "__main__":
    main()
