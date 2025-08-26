from torch.utils.data import DataLoader
from create_dataset import CreateDataset
from loguru import logger
import os
import urllib.request

from config import Settings
from gpt_model import GPT_backbone


def main():
    """
    Main function to create train an LLM
    """

    # import config file
    conf = Settings()

    # Download text to train
    filepath = 'source/the-veredict.txt'
    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
        urllib.request.urlretrieve(url, filepath)
    with open(filepath) as f:
        text = f.read()

    # Create dataset
    dataset = CreateDataset(text, conf.context_lenght, conf.stride)

    # Create dataloader, wrapper for the X and Y Dataset's
    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=conf.shuffle,
        drop_last=conf.drop_last,
        num_workers=conf.num_workers)
    data_iter = iter(dataloader)
    input, target = next(data_iter)

    gpt_model = GPT_backbone(conf)
    target = gpt_model.forward(input)

    logger.info(f"target : {target}")

    # Initiallize Multhiead attention weights and variables
#   mha = MultiHeadAttention(dim_in, dim_out, context_length, 0.0, num_heads)
#   forward = mha.forward(input_embedding)


if __name__ == "__main__":
    main()
