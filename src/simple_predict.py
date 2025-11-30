import re
from collections import defaultdict, Counter

MAX_N = 4  # use up to 4-word context


def load_text():
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        return f.read().lower()


def tokenize(text):
    return re.findall(r"\w+", text)


def build_model(words):
    """
    Build n-gram mappings:
    1-word -> next word counts
    2-words -> next word counts
    3-words -> next word counts
    4-words -> next word counts
    """
    model = defaultdict(Counter)
    for n in range(1, MAX_N + 1):
        for i in range(len(words) - n):
            context = tuple(words[i:i + n])
            next_word = words[i + n]
            model[context][next_word] += 1
    return model


def predict_next_word(model, input_text):
    words = tokenize(input_text)
    if not words:
        return None

    words = words[-MAX_N:]  # keep only last 4 words

    # Try 4-gram → 3-gram → 2-gram → 1-gram
    for n in range(min(MAX_N, len(words)), 0, -1):
        context = tuple(words[-n:])
        if context in model:
            return model[context].most_common(1)[0][0], n

    return None


def main():
    print("Loading training text...")
    text = load_text()
    words = tokenize(text)
    print(f"Total words loaded: {len(words)}")

    print("Building model...")
    model = build_model(words)
    print("Ready! Enter 1 to 4 words. I will predict the next word from training text.")
    print("Type 'exit' to quit.")

    while True:
        user = input("\nEnter text: ").lower().strip()
        if user == "exit":
            print("Bye!")
            break

        result = predict_next_word(model, user)
        if result is None:
            print("No exact match found in training text. Try a different phrase.")
        else:
            next_word, matched = result
            print(f"Next word: {next_word}  (matched last {matched} word(s))")


if __name__ == "__main__":
    main()
