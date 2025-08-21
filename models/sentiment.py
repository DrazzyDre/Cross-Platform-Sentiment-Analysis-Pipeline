
from transformers import pipeline
from typing import Dict, Optional


class SentimentAnalyzer:
    def __init__(self, task: str = "sentiment", model_name: Optional[str] = None):
        """
        task: 'sentiment' for multi-class sentiment, 'emotion' for emotion detection
        model_name: override default model for the task
        """
        if task == "emotion":
            # Multi-class emotion detection (e.g., joy, anger, sadness, etc.)
            self.model_name = model_name or "j-hartmann/emotion-english-distilroberta-base"
            self.classifier = pipeline("text-classification", model=self.model_name, return_all_scores=True)
        else:
            # Multi-class sentiment (positive/neutral/negative)
            self.model_name = model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.classifier = pipeline("sentiment-analysis", model=self.model_name, return_all_scores=True)
        self.tokenizer = self.classifier.tokenizer
        self.task = task

    def predict(self, text: str) -> Dict:
        # Truncate to 512 tokens (model's max length)
        max_length = 512
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length)
        truncated = self.tokenizer.decode(tokens, skip_special_tokens=True)
        results = self.classifier(truncated)[0]
        # results: list of dicts with 'label' and 'score'
        # Find the label with the highest score
        best = max(results, key=lambda x: x['score'])
        # Also return all scores for transparency
        return {
            "label": best["label"],
            "score": float(best["score"]),
            "all_scores": {r["label"]: float(r["score"]) for r in results}
        }

if __name__ == "__main__":
    print("Multi-class sentiment:")
    analyzer = SentimentAnalyzer(task="sentiment")
    sample = "I love this project!"
    print(analyzer.predict(sample))
    print("\nEmotion detection:")
    emotion_analyzer = SentimentAnalyzer(task="emotion")
    print(emotion_analyzer.predict(sample))
