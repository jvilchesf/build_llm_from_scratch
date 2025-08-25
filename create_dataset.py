import tiktoken
import torch
from torch.utils.data import Dataset
from loguru import logger


class CreateDataset(Dataset):
    def __init__(self,
                 text: str,
                 windows_size,
                 stride: int = 4,
                 ):
        """
        Create manually class dataset to be use as input for Dataloader

        Args:
        text: Text file used as input for the process to create toke ids
        windows_size: Define the the number of tokens used to create the sample and target dataset
        stride: Variable that determine the size of jumps for the i variable in the for loop 
        """
        # create ids
        tokenizer = tiktoken.get_encoding("gpt2")
        self.ids = tokenizer.encode(text)
        self.training_sample = []
        self.target_sample = []
        # Create training and target dataset to train the model
        for i in range(0, len(self.ids), stride):

            sample_chunk = self.ids[i: i + windows_size]
            target_chunk = self.ids[i + 1: i + 1 + windows_size]

            self.training_sample.append(torch.tensor(sample_chunk))
            self.target_sample.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.training_sample)

    def __getitem__(self,
                    idx: int):
        return self.training_sample[idx], self.target_sample[idx]
