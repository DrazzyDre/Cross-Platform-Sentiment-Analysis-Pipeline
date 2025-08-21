import pandas as pd

from ingestion.reddit_ingestor import RedditIngestor
from preprocessing.text_preprocessor import TextPreprocessor
from models.sentiment import SentimentAnalyzer
from models.intent import IntentClassifier
from visualization.plots import plot_sentiment_distribution, plot_intent_distribution

# Keyword extraction and summarization imports

from keybert import KeyBERT
from transformers import pipeline as hf_pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# 1. Ingest Reddit data
reddit = RedditIngestor(subreddit="python", limit=50)
posts = reddit.fetch_posts()


# 2. Preprocess text
pre = TextPreprocessor()
for post in posts:
    post['cleaned_text'] = pre.preprocess(post['title'] + ' ' + post['text'])



# 3. Sentiment, emotion, keyword extraction, and summarization
sentiment_model = SentimentAnalyzer(task="sentiment")
emotion_model = SentimentAnalyzer(task="emotion")

kw_model = KeyBERT()
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

# Highlight detection setup (TextRank)
def extract_highlight(text: str) -> str:
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        # Get the most important sentence as highlight
        summary = summarizer(parser.document, sentences_count=1)
        return str(summary[0]) if summary else ''
    except Exception:
        return ''

for post in posts:
    # Sentiment
    sentiment = sentiment_model.predict(post['cleaned_text'])
    post['sentiment'] = sentiment['label']
    post['sentiment_score'] = sentiment['score']
    post['sentiment_all_scores'] = sentiment['all_scores']
    # Emotion
    emotion = emotion_model.predict(post['cleaned_text'])
    post['emotion'] = emotion['label']
    post['emotion_score'] = emotion['score']
    post['emotion_all_scores'] = emotion['all_scores']
    # Keyword extraction (top 5 keywords)
    keywords = kw_model.extract_keywords(post['cleaned_text'], top_n=5)
    post['keywords'] = ', '.join([kw for kw, _ in keywords])
    # Summarization (use original text for better context)
    orig_text = (post.get('title', '') + ' ' + post.get('text', '')).strip()
    if len(orig_text.split()) > 20:
        try:
            summary = summarizer(orig_text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
        except Exception:
            summary = ''
    else:
        summary = orig_text
    post['summary'] = summary
    # Highlight detection (most important sentence)
    post['highlight'] = extract_highlight(orig_text)

# 4. Intent classification (using dummy model for now)
intent_model = IntentClassifier()
# For demo, train on dummy data
texts = ["order pizza", "book flight", "play music", "what's the weather"]
labels = ["food", "travel", "music", "weather"]
intent_model.train(texts, labels)
for post in posts:
    intent = intent_model.predict(post['cleaned_text'])
    post['intent'] = intent['label']
    post['intent_prob'] = max(intent['probabilities'].values())

# 5. Save results
results_df = pd.DataFrame(posts)
results_df.to_csv("results.csv", index=False)

# 6. Visualization
plot_sentiment_distribution(results_df)
plot_intent_distribution(results_df)
