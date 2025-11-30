# Next Word Prediction using N-Gram Language Model

This project predicts the **next word based on 1–4 previous words** using an **N-gram language model**.  
The model learns from `sample.txt` and identifies the most likely next word according to previous word patterns — providing fast and accurate next-word prediction without requiring heavy deep-learning libraries during deployment.

# Features

* Predict next word based on previous words (1 to 4 words input)  
* Works in real-time using FastAPI  
* Local prediction script + Cloud deployed API  
* Lightweight deployment (no TensorFlow required in production)  


 # Project Structure

next-word-prediction/
│
├── data/
│ └── sample.txt
│
├── src/
│ ├── simple_predict.py → Local N-gram prediction
│ ├── train.py → Optional LSTM training script
│ ├── run_predict.py → Optional LSTM prediction script
│ ├── other helper modules…
│
├── deployment/
│ └── api.py → FastAPI deployment script
│
├── saved_models/ → (Optional) stores trained LSTM model
├── requirements.txt
├── README.md
└── .gitignore




# Running the Project Locally

# Install requirements
pip install -r requirements.txt


# Run command-line predictor
python -m src.simple_predict


# Example:
Enter text: machine learning is the
Next word: future


# Live API Deployment

This project is deployed using **FastAPI + Uvicorn** on **Render**.

 **Base API URL**  
`https://next-word-prediction-2zhq.onrender.com`

 **Interactive API Docs (Swagger UI)**  
`https://next-word-prediction-2zhq.onrender.com/docs`

 **Example API Request**  
 https://next-word-prediction-2zhq.onrender.com/predict?q=machine
 learning is the

 
Example response:
```json
{
  "next_word": "future",
  "matched_words": 4
}

 # How N gram model works:

(previous 1–4 words) → next most probable word

# Example learning pattern:

"machine learning is the" → "future"
"deep learning is a" → "subset"


 # Advantages:

Very fast and memory-efficient.
No model loading required.
Deterministic (same result for same input).


## Developer Information:

Name 	                  : Mrunalini  Anup Jadhav
Project Title	          : Next Word Prediction using N-Gram Model
Programming Language  	: Python
Framework (Deployment)	: FastAPI
Hosting Platform	      : Render
College name            : Vivekanand college , kolhapur.
Academic Year	          : BCS TY


Thank you for reviewing this project 🙏.



