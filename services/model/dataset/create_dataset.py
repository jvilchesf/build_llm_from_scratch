import tiktoken
import torch
from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(
        self,
        text: str,
        context_length: int,
        stride: int,
        ratio_split: float,
        split_type: str,
    ):
        """
        Create manually class dataset to be use as input for Dataloader

        Args:
        text: Text file used as input for the process to create toke ids
        context_length: Define the the number of tokens used to create the sample and target dataset
        stride: Variable that determine the size of jumps for the i variable in the for loop
        """
        # create ids
        tokenizer = tiktoken.get_encoding("gpt2")
        self.ids = tokenizer.encode(text)
        self.training_sample = []
        self.target_sample = []
        # Create training and target dataset to train the model
        for i in range(0, len(self.ids), stride):
            sample_chunk = self.ids[i : i + context_length]
            target_chunk = self.ids[i + 1 : i + 1 + context_length]

            self.training_sample.append(torch.tensor(sample_chunk))
            self.target_sample.append(torch.tensor(target_chunk))

        # Calculate ratio split
        total_rows = len(self.training_sample)
        train_rows = int(total_rows * ratio_split)
        if split_type == "train":
            self.training_sample = self.training_sample[:train_rows]
            self.target_sample = self.target_sample[:train_rows]
        elif split_type == "validation":
            self.training_sample = self.training_sample[:train_rows]
            self.target_sample = self.target_sample[:train_rows]

    def __len__(self):
        return len(self.training_sample)

    def __getitem__(self, idx: int):
        return self.training_sample[idx], self.target_sample[idx]
