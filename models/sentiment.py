from transformers import pipeline
from typing import Dict

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.classifier = pipeline("sentiment-analysis", model=model_name)
        self.tokenizer = self.classifier.tokenizer

    def predict(self, text: str) -> Dict:
        # Truncate to 512 tokens (model's max length)
        max_length = 512
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length)
        truncated = self.tokenizer.decode(tokens, skip_special_tokens=True)
        result = self.classifier(truncated)[0]
        return {
            "label": result["label"],
            "score": float(result["score"])
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample = "I love this project!"
    print(analyzer.predict(sample))
