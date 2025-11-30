import os
from tensorflow.keras.callbacks import ModelCheckpoint

from .preprocessing import (
    load_text,
    build_tokenizer,
    create_sequences,
    save_tokenizer,
)
from .model import build_lstm_model


SEQ_LEN = 5  # sequence length used in training and prediction


def main():
    os.makedirs("saved_models", exist_ok=True)

    text = load_text("data/sample.txt")
    if len(text.strip()) == 0:
        raise SystemExit("data/sample.txt is empty. Put sample text there.")

    tokenizer = build_tokenizer(text)
    X, y = create_sequences(tokenizer, text, seq_len=SEQ_LEN)

    if X.size == 0:
        raise SystemExit(
            "Not enough words to create sequences. Add more text to data/sample.txt"
        )

    vocab_size = len(tokenizer.word_index) + 1
    print("vocab_size:", vocab_size, "X shape:", X.shape)

    model = build_lstm_model(vocab_size, SEQ_LEN, embed_dim=64, lstm_units=128)

    checkpoint = ModelCheckpoint("saved_models/model.keras", save_best_only=True)
    model.fit(
        X,
        y,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[checkpoint],
    )

    # final save
    model.save("saved_models/model.keras")
    save_tokenizer(tokenizer, "saved_models/tokenizer.json")
    print("Training finished. Model & tokenizer saved in saved_models/")


if __name__ == "__main__":
    main()
