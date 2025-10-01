from loguru import logger

from services.config import Settings
from services.model.gpt_model import GPT_backbone
from services.model.dataset.create_data_loader import create_data_loader


def main():
    """
    Main function to create train an LLM
    """

    # import config file
    conf = Settings()

    train_dataloader, val_dataloader = create_data_loader(conf)
    data_iter = iter(train_dataloader)
    input, target = next(data_iter)

    gpt_model = GPT_backbone(conf)
    target = gpt_model.forward(input)

    logger.info(f"target : {target}")

    # Initiallize Multhiead attention weights and variables


if __name__ == "__main__":
    main()
