import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load datasets
train_path = train_path = r"C:\Users\ashha\OneDrive\Desktop\Sentiment Analysis\twitter_training.csv"

valid_path = r"C:\Users\ashha\OneDrive\Desktop\Sentiment Analysis\twitter_validation.csv"

train_df = pd.read_csv(train_path, header=None, names=["id", "entity", "label", "tweet"], encoding='utf-8')
valid_df = pd.read_csv(valid_path, header=None, names=["id", "entity", "label", "tweet"], encoding='utf-8')

# Combine datasets
data = pd.concat([train_df, valid_df], ignore_index=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['tweet'] = data['tweet'].astype(str).apply(preprocess_text)

# Encode labels
label_mapping = {"Positive": 1, "Negative": -1, "Neutral": 0, "Hate Speech": -2}
data = data[data['label'].isin(label_mapping.keys())]  # Filter valid labels
data['label'] = data['label'].map(label_mapping)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['label'], test_size=0.2, random_state=42)

# Train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "sentiment_model.pkl")

# Load model
model = joblib.load("sentiment_model.pkl")

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to analyze its sentiment.")

# User input
tweet_input = st.text_area("Enter your tweet:")

if st.button("Analyze Sentiment"):
    processed_input = preprocess_text(tweet_input)
    prediction = model.predict([processed_input])[0]
    sentiment_mapping = {1: "Positive", -1: "Negative", 0: "Neutral", -2: "Hate Speech"}
    st.write(f"Sentiment: {sentiment_mapping[prediction]}")
