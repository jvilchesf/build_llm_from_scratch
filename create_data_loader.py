from create_dataset import CreateDataset
from torch.utils.data import DataLoader


def create_data_loader_v1(text,
                          batch_size: int,
                          windows_size: int,
                          drop_last: bool = True,
                          shuffle: bool = True,
                          num_workers: int = 2,
                          stride: int = 4,
                          ):
    """
    Create data loader function, first it is necessary to create dataset, it is a class with some specific structure
    the dataset class go throught the text and encode or tokenize each word. This class need the function len and getitem to be recognized by the dataloader function

    Args:
    text: str variable with all text that will be processed and used for training
    windows_size: Int variable that specify the number of token to be considered to create the training dataset.
    batch_size: Number of sentences or rows in the training set
    drop_last: Boolean variable defines if the last batch is deleted in the case the number of rows are less than "batch_size"
    shuffle:
    num_workers: int variable define number of cpus used in the dataloader function.
    stride: Int variable to determine size of the jump in the for loop at the moment of creating the dataset.
    """
    # Create dataset X and Y manually
    dataset = CreateDataset(text, windows_size, stride)

    # Create dataloader, wrapper for the X and Y Dataset's
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)

    return dataloader
