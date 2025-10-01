from services.model.dataset.create_data_loader import create_data_loader
from services.config import Settings
from services.model.gpt_model import GPT_backbone
from services.train.train_iteration.train import simple_train

import torch
import tiktoken


def main():
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Import enviromental variables
    conf = Settings()

    # Import dataloaders
    train_dataloader, val_dataloader = create_data_loader(conf)

    # Import model
    model = GPT_backbone(conf)

    # Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    context_start = "The weather in emmaboda is warm"
    train = simple_train(
        model,
        train_dataloader,
        val_dataloader,
        conf,
        optimizer,
        tokenizer,
        context_start,
    )

    print(train)


if __name__ == "__main__":
    main()
