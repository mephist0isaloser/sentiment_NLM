#ffimport the_model_a
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
# Load your CSV data into a pandas DataFrame
data = pd.read_csv('processed_tweets.csv')
if __name__ == "__main__":
    # Create the TF-IDF vectorizer paired with N grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    # Fit and transform the text data
    X_tfidf = vectorizer.fit_transform(data['tweet'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['sentiment'], test_size=0.2, random_state=42)

    # Create and train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

