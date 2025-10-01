from services.model.dataset.create_dataset import CreateDataset
import os
import urllib.request
from torch.utils.data import DataLoader


def create_data_loader(conf):
    # Download text to train
    filepath = "files/the-veredict.txt"
    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
        urllib.request.urlretrieve(url, filepath)
    with open(filepath) as f:
        text = f.read()

    # Create dataset
    train_dataset = CreateDataset(
        text, conf.context_length, conf.stride, ratio_split=0.8, split_type="train"
    )

    val_dataset = CreateDataset(
        text,
        conf.context_length,
        conf.stride,
        ratio_split=0.2,
        split_type="validation",
    )

    # Create dataloader, wrapper for the X and Y Dataset's
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=conf.shuffle,
        drop_last=conf.drop_last,
        num_workers=conf.num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.batch_size,
        shuffle=conf.shuffle,
        drop_last=conf.drop_last,
        num_workers=conf.num_workers,
    )

    return train_dataloader, val_dataloader
