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
from bertopic import BERTopic
import spacy
from fairlearn.metrics import demographic_parity_difference



# --- FLEXIBLE INGESTION CLI ---
import sys

def get_input(prompt, default=None, cast_func=None):
    val = input(f"{prompt} [{default}]: ")
    if not val and default is not None:
        return default
    if cast_func:
        try:
            return cast_func(val)
        except Exception:
            print(f"Invalid input. Using default: {default}")
            return default
    return val


print("\nReddit Ingestion Configuration:")
subreddits_str = get_input("Subreddit(s) to analyze (comma-separated)", default="python")
subreddits = [s.strip() for s in subreddits_str.split(",") if s.strip()]
mode = get_input("Ingestion mode ('top', 'keyword', 'both')", default="top").lower()
post_limit = get_input("Number of posts to fetch (per subreddit, per mode)", default=50, cast_func=int)
keywords = []
min_score = 0
if mode in ("keyword", "both"):
    kw_str = get_input("Enter 2-5 keywords (comma-separated)", default="pandas,dataframe,plot")
    keywords = [k.strip() for k in kw_str.split(",") if k.strip()]
    if not (2 <= len(keywords) <= 5):
        print("Warning: 2-5 keywords recommended. Using all provided.")
    min_score = get_input("Minimum score (engagement) for keyword search", default=10, cast_func=int)

# Multi-subreddit ingestion
all_posts = []
for subreddit in subreddits:
    reddit = RedditIngestor(subreddit=subreddit, limit=post_limit)
    if mode == 'top':
        posts = reddit.fetch_posts(mode='top', limit=post_limit)
    elif mode == 'keyword':
        posts = reddit.fetch_posts(mode='keyword', keywords=keywords, limit=post_limit, min_score=min_score)
    elif mode == 'both':
        posts = reddit.fetch_posts(mode='both', keywords=keywords, limit=post_limit, min_score=min_score)
    else:
        print(f"Invalid mode for subreddit {subreddit}. Skipping.")
        continue
    all_posts.extend(posts)

# Deduplicate posts by id across all subreddits
seen_ids = set()
posts = []
for post in all_posts:
    if post['id'] not in seen_ids:
        posts.append(post)
        seen_ids.add(post['id'])


# 2. Preprocess text
pre = TextPreprocessor()
for post in posts:
    post['cleaned_text'] = pre.preprocess(post['title'] + ' ' + post['text'])



# 3. Sentiment, emotion, keyword extraction, and summarization
# 3. Sentiment, emotion, keyword extraction, and summarization

sentiment_model = SentimentAnalyzer(task="sentiment")
emotion_model = SentimentAnalyzer(task="emotion")
kw_model = KeyBERT()
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

# Topic modeling setup
topic_model = BERTopic(verbose=False)

# Entity linking setup
nlp = spacy.load("en_core_web_sm")

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


texts_for_topic = []
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
    word_count = len(orig_text.split())
    min_length = 15
    # Set max_length to 80% of input length, but at most 60, and always > min_length
    max_length = min(60, max(min_length+1, int(word_count * 0.8)))
    if word_count > min_length and max_length > min_length:
        try:
            summary = summarizer(orig_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        except Exception:
            summary = ''
    else:
        summary = orig_text
    post['summary'] = summary
    # Highlight detection (most important sentence)
    post['highlight'] = extract_highlight(orig_text)
    # For topic modeling: only add non-empty strings
    cleaned = post['cleaned_text']
    if isinstance(cleaned, str) and cleaned.strip():
        texts_for_topic.append(cleaned)
    # Entity linking (NER)
    doc = nlp(orig_text)
    post['entities'] = ', '.join([f"{ent.text} ({ent.label_})" for ent in doc.ents])


# 4. Improved intent classification (multi-class, e.g., ask/feedback/rant)
intent_model = IntentClassifier()
# Example: train on more realistic intent classes
intent_texts = [
    "How do I install this?", "Can someone help me?", "This feature is great!", "I hate this bug.",
    "Why is this not working?", "Thanks for the update!", "This is so frustrating.", "Feedback: UI is confusing."
]
intent_labels = [
    "ask", "ask", "feedback", "rant",
    "ask", "feedback", "rant", "feedback"
]
intent_model.train(intent_texts, intent_labels)
for post in posts:
    intent = intent_model.predict(post['cleaned_text'])
    post['intent'] = intent['label']
    post['intent_prob'] = max(intent['probabilities'].values())


# 5. Topic modeling (BERTopic)
topics, _ = topic_model.fit_transform(texts_for_topic)
for i, post in enumerate(posts):
    post['topic'] = topics[i]

# 6. Save results
results_df = pd.DataFrame(posts)
results_df.to_csv("results.csv", index=False)


# 7. Visualization
plot_sentiment_distribution(results_df)
plot_intent_distribution(results_df)

# 8. Bias handling (Fairlearn demo: demographic parity on sentiment by author)
try:
    if 'author' in results_df.columns:
        # For demo, treat sentiment POSITIVE as 1, else 0
        y_true = (results_df['sentiment'] == 'positive').astype(int)
        groups = results_df['author']
        dp = demographic_parity_difference(y_true, y_pred=y_true, sensitive_features=groups)
        print(f"Demographic parity difference (sentiment vs. author): {dp}")
except Exception as e:
    print(f"Fairness evaluation error: {e}")
