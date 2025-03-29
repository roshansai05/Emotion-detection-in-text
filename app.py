import os
import sys
import ssl
import joblib
import re
import nltk
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Fix SSL certificate issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure necessary NLTK resources are available
nltk_resources = ["stopwords", "punkt", "wordnet"]
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Check if model files exist
model_path = "emotion_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"
label_encoder_path = "label_encoder.pkl"

if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path]):
    raise FileNotFoundError("One or more model files are missing. Ensure .pkl files are correctly generated.")

# Load trained model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    text = request.form.get('text')
    if not text:
        return render_template('predict.html', emotion="No text provided")
    
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)
    emotion = label_encoder.inverse_transform(prediction)[0]

    return render_template('predict.html', emotion=emotion)

if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    port = int(os.getenv("PORT", 5500)) 
    app.run(port=port, debug=debug_mode)