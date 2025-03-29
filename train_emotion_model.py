import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Using SVM instead of Logistic Regression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Load dataset
df = pd.read_csv("preprocessed_tweet_emotions.csv")

# Drop missing values
df.dropna(subset=['clean_text'], inplace=True)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment'])

# Balance the dataset (reduce majority, increase minority)
majority_classes = ['neutral', 'worry']
minority_classes = ['anger', 'boredom']

# Separate majority and minority classes
df_majority = df[df['sentiment'].isin(majority_classes)]
df_minority = df[df['sentiment'].isin(minority_classes)]

# Undersample majority classes (reduce their size)
df_majority = df_majority.sample(n=3000, random_state=42)

# Oversample minority classes (increase their size)
df_minority_upsampled = df_minority.sample(n=1000, replace=True, random_state=42)

# Combine the balanced dataset
df_balanced = pd.concat([df_majority, df[df['sentiment'].isin(set(df['sentiment']) - set(majority_classes + minority_classes))], df_minority_upsampled])

# TF-IDF Vectorization with Bigrams (captures more context)
tfidf_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))  # Unigrams + Bigrams
X = tfidf_vectorizer.fit_transform(df_balanced['clean_text'].astype(str)).toarray()
y = df_balanced['label']

# Split dataset (More training data: 95% train, 5% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Train Support Vector Machine (SVM) Model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f' New Model Accuracy: {accuracy * 100:.2f}%')

# Save Model, Vectorizer, and Label Encoder
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Balanced Model and Vectorizer Saved Successfully!")
 