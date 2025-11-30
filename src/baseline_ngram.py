# src/baseline_ngram.py
from collections import defaultdict, Counter
import re

def build_trigram_model(text):
    words = re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
    trigram = defaultdict(Counter)
    for i in range(len(words)-2):
        context = (words[i], words[i+1])
        trigram[context][words[i+2]] += 1
    return trigram

def predict_trigram(trigram, seed):
    seed_words = re.sub(r'[^a-z0-9\s]', '', seed.lower()).split()
    if len(seed_words) < 2:
        return None
    context = (seed_words[-2], seed_words[-1])
    if context not in trigram:
        return None
    return trigram[context].most_common(1)[0][0]

if _name_ == "_main_":
    import sys
    path = "data/sample.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    trigram = build_trigram_model(text)
    seed = "this is a"
    print("Trigram prediction for seed:", seed, "->", predict_trigram(trigram, seed))

