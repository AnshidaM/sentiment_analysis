# ================================
# IMPORTS
# ================================

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import emoji

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# ================================
# APP INIT
# ================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# LOAD MODEL
# ================================
data = joblib.load("sentiment_model.pkl")
model = data["model"]
vectorizer = data["vectorizer"]

# ================================
# NLP SETUP
# ================================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

neutral_words = ["okay", "fine", "average", "normal", "not bad", "not great"]

def boost_neutral(text):
    for word in neutral_words:
        if re.search(rf"\b{word}\b", text):
            return text + " neutral"
    return text

def preprocess(text):
    text = boost_neutral(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)
# ================================
# EMOJI MODEL
# ================================
transformer_model = None

def get_transformer():
    global transformer_model
    if transformer_model is None:
        from transformers import pipeline
        transformer_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return transformer_model



def contains_emoji(text):
    return any(char in emoji.EMOJI_DATA for char in text)

# ================================
# REQUEST SCHEMA
# ================================
class TextInput(BaseModel):
    text: str

# ================================
# ROUTES
# ================================
@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: TextInput):

    text_input = data.text

    if not text_input.strip():
        return {"sentiment": "neutral"}

    # emoji → transformer
    if contains_emoji(text_input):
        result = transformer_model(text_input)[0]
        label = result["label"].lower()

        if "pos" in label:
            return {"sentiment": "positive"}
        elif "neg" in label:
            return {"sentiment": "negative"}
        else:
            return {"sentiment": "neutral"}

    # SVM model
    processed = preprocess(text_input)
    text_vec = vectorizer.transform([processed])

    if text_vec.nnz == 0:
        return {"sentiment": "neutral"}

    prediction = model.predict(text_vec)[0]

    return {"sentiment": prediction}