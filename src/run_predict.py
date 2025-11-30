from tensorflow.keras.models import load_model
from .predict import load_tokenizer, predict_next_word

SEQ_LEN = 5  # must match SEQ_LEN used in train.py


def main():
    print("Loading model and tokenizer...")
    model = load_model("saved_models/model.h5")
    tokenizer = load_tokenizer("saved_models/tokenizer.json")
    print("Ready! Type some words and I will predict the next word.")
    print("Type 'exit' to quit.")

    while True:
        text = input("\nEnter text: ")

        if text.strip().lower() == "exit":
            print("Bye!")
            break

        next_word = predict_next_word(model, tokenizer, text, seq_len=SEQ_LEN)

        if next_word is None:
            print("I couldn't predict. Maybe the input is empty or words are not in training text.")
        else:
            print("Predicted next word:", next_word)


if __name__ == "__main__":
    print("__name__ is:", repr(__name__))
    main()
