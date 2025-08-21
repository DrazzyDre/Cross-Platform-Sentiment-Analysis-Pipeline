
# Cross-Platform Sentiment Analysis Pipeline

This project provides a modular, scalable, and production-ready pipeline for sentiment, emotion, intent, and topic analysis on social media data, starting with Reddit. The codebase is organized for easy extensibility to other platforms (e.g., Twitter, LinkedIn).

---

## Features

- **Flexible Reddit Ingestion:**
   - Fetch posts from one or multiple subreddits (multi-subreddit support).
   - Ingest by top posts, keyword search, or both (with deduplication).
   - Filter by minimum engagement (score).
   - Fetch post metadata and comments.
- **Text Preprocessing:**
   - Clean, normalize, and lemmatize text for model input and readability.
- **Sentiment & Emotion Analysis:**
   - Multi-class sentiment (positive/neutral/negative) and emotion detection (joy, anger, etc.) using state-of-the-art models.
- **Intent Classification:**
   - Classify user intent (ask, feedback, rant, etc.) with a trainable model.
- **Keyword Extraction & Summarization:**
   - Extract top keywords (KeyBERT) and generate summaries (transformers).
- **Highlight & Entity Detection:**
   - Extract highlights (TextRank) and named entities (spaCy NER).
- **Topic Modeling:**
   - Discover topics in posts using BERTopic.
- **Bias & Fairness Metrics:**
   - Evaluate demographic parity (Fairlearn) on sentiment by author.
- **Visualization:**
   - Generate sentiment and intent distribution plots (matplotlib/seaborn).
- **Extensible:**
   - Modular design for easy addition of new platforms, models, or analytics.

---

## Tech Stack

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- transformers (Hugging Face)
- scikit-learn
- praw (Reddit API)
- streamlit (optional dashboard)
- emoji, nltk, torch, python-dotenv, keybert, sumy, bertopic, spacy, fairlearn

---

## Directory Structure

- `ingestion/` — Data collection modules (Reddit, future platforms)
- `preprocessing/` — Text cleaning and normalization
- `models/` — Sentiment, emotion, and intent analysis
- `visualization/` — Plots and dashboard
- `main.py` — Pipeline entry point (CLI)
- `requirements.txt` — All dependencies

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd Cross-Platform-Sentiment-Analysis-Pipeline
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Configure Reddit API credentials:**
    - Copy your credentials into `ingestion/reddit_config.py`:
       ```python
       REDDIT_CLIENT_ID = "your_client_id"
       REDDIT_CLIENT_SECRET = "your_client_secret"
       REDDIT_USER_AGENT = "your_user_agent"
       ```
    - [Create a Reddit app](https://www.reddit.com/prefs/apps) if you don't have credentials.

---

## Usage (CLI)

Run the pipeline interactively:

```bash
python main.py
```

You will be prompted for:

| Prompt                                         | Example Input                        | Description                                                      |
|------------------------------------------------|--------------------------------------|------------------------------------------------------------------|
| Subreddit(s) to analyze (comma-separated)      | immigration,visas,CanadaVisa         | One or more subreddit names (no `/r/` or URL)                    |
| Ingestion mode ('top', 'keyword', 'both')      | keyword                              | 'top' (top posts), 'keyword' (search), or 'both'                 |
| Number of posts to fetch (per subreddit, mode) | 50                                   | How many posts to fetch per subreddit per mode                   |
| Enter 2-5 keywords (comma-separated)           | visa,student visa,education visa     | Only for 'keyword'/'both' modes; posts must match any keyword    |
| Minimum score (engagement) for keyword search  | 10                                   | Only posts with at least this score are included                 |

**Example:**

```
Subreddit(s) to analyze (comma-separated) [python]: visas,immigration
Ingestion mode ('top', 'keyword', 'both') [top]: keyword
Number of posts to fetch (per subreddit, per mode) [50]: 50
Enter 2-5 keywords (comma-separated) [pandas,dataframe,plot]: visa,student visa,education visa
Minimum score (engagement) for keyword search [10]: 6
```

---

## Output

- **CSV:** Results are saved as `results.csv` with all metadata and analysis columns.
- **Plots:** Sentiment and intent distribution plots are shown after analysis.
- **Sample columns in CSV:**
   - id, author, timestamp, score, subreddit, title, text, cleaned_text
   - sentiment, sentiment_score, emotion, keywords, summary, highlight, entities, intent, topic

---

## Customization & Extending

- **Add new platforms:** Create a new ingestor in `ingestion/` and update `main.py`.
- **Add new analytics:** Add new models or metrics in `models/` and update the pipeline.
- **Change models:** Swap out Hugging Face model names in `main.py` or `models/`.
- **Visualization:** Add new plots in `visualization/plots.py` or build a Streamlit dashboard.

---

## Troubleshooting

- **Reddit API errors:** Check credentials in `ingestion/reddit_config.py`.
- **No posts returned:** Try lowering the minimum score or using more general keywords.
- **Model warnings:** Some warnings (e.g., about max_length) are informational and do not affect results.
- **BERTopic errors:** Ensure all texts are non-empty strings (pipeline handles this automatically).

---

## Example Outputs

- Example CSVs and plots will be generated after running the pipeline.
- See `results.csv` for a sample output.

---

## License

MIT License. See `LICENSE` file for details.

---

For detailed usage and examples, see the docstrings in each module and the code comments in `main.py`.
