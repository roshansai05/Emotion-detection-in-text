import joblib

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Model, Vectorizer, and Label Encoder loaded successfully!")
