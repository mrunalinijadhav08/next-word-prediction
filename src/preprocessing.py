import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_tokenizer(text, num_words=None):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    return tokenizer


def create_sequences(tokenizer, text, seq_len=5):
    # convert text to sequence of integers
    sequences = tokenizer.texts_to_sequences([text])[0]

    X = []
    y = []

    # create sliding window sequences
    for i in range(seq_len, len(sequences)):
        X.append(sequences[i - seq_len : i])
        y.append(sequences[i])

    if len(X) == 0:
        return np.array([]), np.array([])

    X = np.array(X)
    y = np.array(y)

    # ensure shape (num_samples, seq_len)
    X = pad_sequences(X, maxlen=seq_len, padding="pre", truncating="pre")

    return X, y


def save_tokenizer(tokenizer, path):
    data = tokenizer.to_json()
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
