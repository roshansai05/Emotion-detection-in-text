# Emotion-detection-in-text
This project focuses on detecting human emotions from text using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify text into different emotions such as happiness, sadness, anger, surprise, fear, and more. The application provides an interactive full-stack web interface where users can input text and receive real-time emotion predictions.

The project includes a Flask-based backend that processes user input and returns predicted emotions. A machine learning model is trained from scratch using a well-curated dataset and is integrated into the backend using a pre-trained model file (.pkl). The frontend is designed to be user-friendly, with smooth navigation and responsive elements.

The frontend is built using HTML, CSS, and JavaScript with Bootstrap for responsive design. AJAX is used to send user input to the backend asynchronously. The backend is implemented using Flask, which handles HTTP requests, processes text input, and returns predictions. Pandas and NumPy are used for text preprocessing, while Scikit-learn and TensorFlow/PyTorch are used for training the model.

The dataset consists of text samples labeled with different emotions. The data is preprocessed by removing stopwords, punctuation, and converting text to lowercase. The text is then converted into numerical features using TF-IDF or word embeddings. A machine learning LSTM model is trained in Google Colab and saved as a .pkl file. This model is then integrated into the Flask backend for real-time predictions.

When a user enters text in the frontend, it is sent to the Flask API, which processes the text and returns the detected emotion. The result is then displayed on the webpage dynamically. The application is tested for accuracy, speed, and usability before deployment.

To run the project, first clone the repository and install the required dependencies.
Create a folder templates and under the folder place index.html and predict.html files for smooth execution
