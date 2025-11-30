import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    tokenizer = tokenizer_from_json(data)
    return tokenizer


def predict_next_word(model, tokenizer, input_text, seq_len=5):
    text = input_text.strip()
    if not text:
        return None

    # convert input text to sequence of integers
    encoded = tokenizer.texts_to_sequences([text])[0]
    if len(encoded) == 0:
        # none of the words are in the vocabulary
        return None

    # pad to the same length used during training
    encoded = pad_sequences([encoded], maxlen=seq_len, padding="pre")

    # predict probabilities
    y_hat = model.predict(encoded, verbose=0)[0]
    predicted_index = int(np.argmax(y_hat))

    # find the word that has this index
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return None
