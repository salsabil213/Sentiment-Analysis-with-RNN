import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 50  # same as used in training

# Streamlit UI
st.title("Tweet Sentiment Analyzer")

user_input = st.text_area("Enter a tweet")

if st.button("Analyze"):
    seq = tokenizer.texts_to_sequences([user_input])
    pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(pad)[0][0]
    
    sentiment = "Positive" if pred > 0.5 else "Negative"
    st.write(f"**Sentiment:** {sentiment} ({pred:.2f})")
