from services.load_weights.model_weights import get_model
from services.train.generate_text.generate import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

import chainlit

model, conf, tokenizer = get_model()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(message.content, tokenizer),
        max_new_tokens=50,
        context_size=conf.context_length,
        top_k=1,
        temperature=0.0,
    )

    text = token_ids_to_text(token_ids, tokenizer)

    await chainlit.Message(content=f"{text}").send()
