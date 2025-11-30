from fastapi import FastAPI
import re
from collections import defaultdict, Counter

app = FastAPI()

MAX_N = 4

# Load and prepare model only once
def load_text():
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        return f.read().lower()

def tokenize(text):
    return re.findall(r"\w+", text)

def build_model(words):
    model = defaultdict(Counter)
    for n in range(1, MAX_N + 1):
        for i in range(len(words) - n):
            context = tuple(words[i:i + n])
            next_word = words[i + n]
            model[context][next_word] += 1
    return model

text = load_text()
words = tokenize(text)
model = build_model(words)


@app.get("/predict")
def predict(q: str):
    words = tokenize(q.lower())
    if not words:
        return {"next_word": None}

    for n in range(min(MAX_N, len(words)), 0, -1):
        context = tuple(words[-n:])
        if context in model:
            prediction = model[context].most_common(1)[0][0]
            return {"next_word": prediction, "matched_words": n}

    return {"next_word": None}
