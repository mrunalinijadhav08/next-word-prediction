from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def build_lstm_model(vocab_size, seq_len, embed_dim=64, lstm_units=256):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
        )
    )
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model
