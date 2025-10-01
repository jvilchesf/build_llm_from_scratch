from services.train.loss_calculation.loss import calc_loss_batch
from services.train.evaluation.evaluate_model import (
    evaluate_model,
    generate_and_print_sample,
)

from loguru import logger


def simple_train(
    model, train_dataloader, val_dataloader, conf, optimizer, tokenizer, context_start
):
    """
    process to train model based:
    1. import dataloaders, train and validation
    2. define number of epochs, 1 epoch is a complete iteration over the dataset
    3. clean weihts to start new epoch
    4. Iterate over loader
    5. iterate over batches
    6. batch forward
        6.1 Receive batch with x number of samples
        6.2 Run The model over the batch
        6.3 Return results
    7. calculate lost
    8. Update weights
    9. print samples to check how the model is doing
    """
    train_loss_array, val_loss_array = [], []
    token_seen, global_step = 0, -1

    for epoch in range(conf.num_epochs):
        # Set model in training mode
        model.train()

        logger.info(f"Start training process, epoch: {epoch}")

        for i, (input_batch, target_batch) in enumerate(train_dataloader):
            # clean gradiient from pass iterations
            optimizer.zero_grad()

            # loss
            loss = calc_loss_batch(model, input_batch, target_batch)

            # gradients calculate
            loss.backward()

            # Update model weights
            optimizer.step()

            token_seen += input_batch.numel()
            global_step += 1

            # Evaluate model
            if (global_step % conf.freq_eva) == 0:
                # Evaluate train and validation dataloader
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, conf.num_batch_evaluate
                )

                train_loss_array.append(train_loss)
                val_loss_array.append(val_loss)
                logger.info(
                    f"global_steps: {global_step}, train loss: {train_loss}, validation loss: {val_loss}"
                )

                generate_and_print_sample(model, tokenizer, context_start)
