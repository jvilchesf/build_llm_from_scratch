import torch
from loguru import logger


def calc_loss_batch(model, input_batch, target_batch):
    """
    Each batch = [batch_size x context_length]
    1. send input to model
    2. evaluate result with cross_entropy
    """

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def calc_loss_dataloader(model, dataloader, num_batches_evaluate):
    """
    Caluculate loss for the all dataloader
    """

    loss = 0
    # define for how many batches loss calculation
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches_evaluate is None:
        num_batches_evaluate = len(dataloader)
    else:
        num_batches_evaluate = min(num_batches_evaluate, len(dataloader))

    i = 0
    # Iterate over batches
    for i, (train_batch, target_batch) in enumerate(dataloader):
        # return if iterations are bigger that batch numbers defined
        if i < num_batches_evaluate:
            # calculate loss
            loss_batch = calc_loss_batch(model, train_batch, target_batch)

            # Sum losses
            loss += loss_batch.item()

        else:
            break
    return loss / num_batches_evaluate
