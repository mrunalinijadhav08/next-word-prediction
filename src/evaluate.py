# src/evaluate.py
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessing import load_tokenizer, load_text, create_sequences
import os

def main():
    if not os.path.exists("saved_models/model.h5"):
        raise SystemExit("saved_models/model.h5 not found. Train the model first.")
    model = load_model("saved_models/model.h5")
    tokenizer = load_tokenizer("saved_models/tokenizer.json")
    text = load_text("data/sample.txt")
    X, y = create_sequences(tokenizer, text, seq_len=5)
    if X.size == 0:
        raise SystemExit("No evaluation sequences found.")
    preds = model.predict(X, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    accuracy = np.mean(pred_indices == y)
    print(f"Evaluation accuracy on training text (approx): {accuracy:.4f}")

if _name_ == "_main_":
    main()
