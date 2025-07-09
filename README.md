Date: June 2023

Overview
This project builds a machine learning pipeline to detect fake news articles using Natural Language Processing (NLP) techniques. By combining the article title and body text, cleaning the data, and applying TF-IDF vectorization, the model classifies news as Fake or Real with high accuracy.

Dataset
The dataset consists of two CSV files:

Fake.csv: Contains fake news articles

True.csv: Contains real news articles

Note: Due to dataset size, the data files are not included here. You can download the dataset from Kaggle Fake News Dataset.

Approach
Data Loading: Loaded and combined the fake and real news datasets.

Preprocessing:

Combined title and article text.

Converted text to lowercase, removed punctuation, numbers, and stopwords.

Lemmatized words for better normalization.

Feature Extraction: Applied TF-IDF vectorization to convert text into numerical features.

Model Training: Trained a Logistic Regression classifier on the vectorized features.

Evaluation:

Achieved ~99% accuracy and ROC AUC.

Detailed evaluation includes precision, recall, F1-score, and confusion matrix.

Results
Metric	Score
Accuracy	0.99
Precision (Fake)	0.99
Recall (Fake)	0.99
F1-Score (Fake)	0.99
ROC AUC	0.99

Usage
Clone the repo

Download the dataset from the Kaggle link above

Run the fake_news_detector.ipynb notebook step-by-step

Modify and experiment with different models or preprocessing as needed

Tools & Libraries
Python 3

Pandas

Numpy

NLTK

Scikit-learn

Jupyter Notebook

