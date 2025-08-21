import pandas as pd
from ingestion.reddit_ingestor import RedditIngestor
from preprocessing.text_preprocessor import TextPreprocessor
from models.sentiment import SentimentAnalyzer
from models.intent import IntentClassifier
from visualization.plots import plot_sentiment_distribution, plot_intent_distribution

# 1. Ingest Reddit data
reddit = RedditIngestor(subreddit="python", limit=50)
posts = reddit.fetch_posts()

# 2. Preprocess text
pre = TextPreprocessor()
for post in posts:
    post['cleaned_text'] = pre.preprocess(post['title'] + ' ' + post['text'])

# 3. Sentiment analysis
sentiment_model = SentimentAnalyzer()
for post in posts:
    sentiment = sentiment_model.predict(post['cleaned_text'])
    post['sentiment'] = sentiment['label']
    post['sentiment_score'] = sentiment['score']

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
