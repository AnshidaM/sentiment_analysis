# ================================
# IMPORTS
# ================================
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import emoji
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from fastapi.middleware.cors import CORSMiddleware

# ================================
# NLTK (keep for local, move to Docker later)
# ================================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# ================================
# PREPROCESSING (MATCH TRAINING)
# ================================
def normalize_contractions(text):
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text



def enrich_with_emoji(text):
    em = emoji.demojize(text)
    em = em.replace(":", " ")
    return text + " " + em + " " + em   # 👈 double boost

def preprocess(text):
    text = normalize_contractions(text)   # 👈 add this
    text = enrich_with_emoji(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)
# ================================
# CONTEXT RULE (IMPORTANT)
# ================================
def context_override(text, prediction):
    text_lower = text.lower()

    if "not good" in text_lower or "not great" in text_lower:
        return "negative"

    if "but" in text_lower:
        parts = text_lower.split("but")
        if len(parts) > 1:
            return model.predict(
                vectorizer.transform([preprocess(parts[-1])])
            )[0]

    return prediction

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

    # ================================
    # SVM MODEL (PRIMARY)
    # ================================
    processed = preprocess(text_input)
    text_vec = vectorizer.transform([processed])

    if text_vec.nnz == 0:
        return {"sentiment": "neutral"}

    prediction = model.predict(text_vec)[0]

    # apply context rule
    final_pred = context_override(text_input, prediction)

    return {"sentiment": final_pred}