import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = emoji.replace_emoji(text, replace='')  # Remove emojis
        text = re.sub(r'@[\w_]+', '', text)  # Remove mentions
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

if __name__ == "__main__":
    pre = TextPreprocessor()
    sample = "Check out https://github.com! @user 😊 This is a test, running at 100%."
    print("Original:", sample)
    print("Cleaned:", pre.preprocess(sample))
