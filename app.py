import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('sentiment_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review and find out if it's Positive or Negative!")

review = st.text_area("Movie Review:", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=200)
        prediction = model.predict(padded)[0][0]
        sentiment = "😊 Positive" if prediction > 0.5 else "☹️ Negative"
        st.success(f"Sentiment: **{sentiment}**")
