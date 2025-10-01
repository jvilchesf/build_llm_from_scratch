from services.train.loss_calculation.loss import calc_loss_dataloader
from services.train.generate_text.generate import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)

import torch


def evaluate_model(
    model,
    train_dataloader,
    val_dataloader,
    num_batch_evaluate,
):
    """
    calculate loss for train and val dataloader
    """

    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_dataloader(model, train_dataloader, num_batch_evaluate)
        val_loss = calc_loss_dataloader(model, val_dataloader, num_batch_evaluate)

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, context_start):
    model.eval()
    # Get context size from embedings in model
    # context size represent the number of words ni a phrase
    context_size = model.pos_emb.weight.shape[0]
    # Context start is a text that the model will complement during training
    # Transform context start into ids
    encoded = text_to_token_ids(context_start, tokenizer)
    with torch.no_grad():
        # Send ids to the model for it to append a new word to the context_start text
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

    # Transform ids back into text
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", ""))
    model.train()
