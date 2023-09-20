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

@app.route('/predict_tags', methods=['POST'])
def predict_tags_api():
    try:
        data = request.get_json()
        text = data['text']

        tokens = preprocess_text(text)

        X_vectorized = vectorizer.transform([' '.join(tokens)])

        predictions = clf.predict(X_vectorized)

	predicted_tags = mlb.inverse_transform(predictions)

        return jsonify({'tags': predicted_tags[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
