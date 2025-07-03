import subprocess
import streamlit as st
import re
import tensorflow as tf
import unicodedata
import spacy
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

# Use /tmp for NLTK data (writable in Hugging Face Spaces)
nltk_data_dir = "/tmp/nltk_data"
nltk.data.path.append(nltk_data_dir)

# Download the stopwords and punkt resources
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)

# Download spaCy model jika belum ada
try:
    spacy.load("en_core_web_sm")
except:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load model & vectorizer & encoder
model = tf.keras.models.load_model("src/model_sentiment_ann.h5")

# Load model ANN terbaik
# model = load_model('model_sentiment_ann.h5')
vectorizer_model = tf.keras.models.load_model("src/text_vectorizer_model.keras")

with open("src/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)

    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])

    text = re.sub(r'\bmln\b|\bmn\b|\bmillion\b', 'million', text)
    text = re.sub(r'\beur\b|\beuro\b', 'euro', text)

    return text

# Streamlit UI
st.title("Financial Sentiment Analysis")

user_input = st.text_area("Enter a financial sentence to analyze:")

if st.button("Analyze"):
    if user_input:
        cleaned = preprocess_text(user_input)
        sequence = vectorizer_model(tf.constant([cleaned]))
        probs = model.predict(sequence)
        pred_label = np.argmax(probs, axis=1)
        sentiment = label_encoder.inverse_transform(pred_label)[0]
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a sentence.")