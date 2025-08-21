from transformers import pipeline
from typing import Dict

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.classifier = pipeline("sentiment-analysis", model=model_name)

    def predict(self, text: str) -> Dict:
        result = self.classifier(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"])
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample = "I love this project!"
    print(analyzer.predict(sample))
