# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import re

app = Flask(__name__)

def preprocess_text(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

vectorizer = joblib.load('vectorizer.pkl')
clf = joblib.load('clf.pkl')
mlb = joblib.load('mlb.pkl')

vectorizer_lda = joblib.load('vectorizer_lda.pkl')
lda = joblib.load('lda.pkl')

"""@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        tokens = preprocess_text(text)
        X_vectorized = vectorizer.transform([' '.join(tokens)])
        predictions = clf.predict(X_vectorized)
        predicted_tags = mlb.inverse_transform(predictions)
        return jsonify({'tags': predicted_tags[0]})
    except Exception as e:
        return jsonify({'error': str(e)})"""


@app.route('/predict_lda', methods=['POST'])
def predict_lda():
    try:
        data = request.get_json()
        text = data['text']
        tokens = preprocess_text(text)
        X_vectorized = vectorizer_lda.transform([' '.join(tokens)])
        prediction = lda.transform(X_vectorized)
        return jsonify({'Prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
