import re


class Tokenizer:
    def __init__(self,
        vocab: dict):

        self.str_to_int: dict = vocab
        self.int_to_string: dict = {k: v for k, v in vocab.items()}

    def encode(self,
               text: str):
        """
        Takes full text, and based on the vocabulary befored built, a list with full tokenized words is returned
        """
        text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [word.strip() for word in text if word.strip()]
        encoded_text = [self.str_to_int[word] for word in preprocessed]
        return encoded_text

    def decode(self,
               ids: list):
        """
        Transform id lists back into text
        """
        decoded_text = " ".join([self.int_to_string[id] for id in ids])
        decoded_text = re.sub(r'\s+([,.?!"()\'])', r'\1', decoded_text)
        return decoded_text  
    # Create dictionary
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_data)
    preprocessed = [word.strip() for word in preprocessed if word.strip()]
    all_words = sorted(set(preprocessed))
    vocabulary = {word: index for index, word in enumerate(all_words)}

    tokenizer = Tokenizer(vocabulary)
    encoded_text = tokenizer.encode(raw_data)

    print(encoded_text)


