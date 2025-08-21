import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List, Dict
import joblib

class IntentClassifier:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(model_path + '.vec')
        else:
            self.model = LogisticRegression(max_iter=200)
            self.vectorizer = TfidfVectorizer()

    def train(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text: str) -> Dict:
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        label = self.model.classes_[proba.argmax()]
        return {
            "label": label,
            "probabilities": dict(zip(self.model.classes_, proba))
        }

    def save(self, model_path: str):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, model_path + '.vec')

if __name__ == "__main__":
    # Example usage with dummy data
    texts = ["order pizza", "book flight", "play music", "what's the weather"]
    labels = ["food", "travel", "music", "weather"]
    clf = IntentClassifier()
    clf.train(texts, labels)
    print(clf.predict("order food"))
    clf.save("intent_model.joblib")
