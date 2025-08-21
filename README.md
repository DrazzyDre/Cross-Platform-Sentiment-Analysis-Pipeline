# Cross-Platform Sentiment Analysis Pipeline

This project provides a modular, scalable, and production-ready pipeline for sentiment and intent analysis on social media data, starting with Reddit. The codebase is organized for easy extensibility to other platforms (e.g., Twitter, LinkedIn).

## Features
- **Data Ingestion:** Fetch Reddit posts/comments with metadata using the official API.
- **Preprocessing:** Clean and normalize text for analysis.
- **Sentiment & Intent Analysis:** Use pre-trained/fine-tuned models for sentiment and intent classification.
- **Data Storage:** Save results in CSV or SQLite DB.
- **Visualization:** Generate plots and (optionally) an interactive Streamlit dashboard.

## Tech Stack
- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- transformers (Hugging Face)
- scikit-learn
- praw (Reddit API)
- streamlit (optional dashboard)

## Directory Structure
- `ingestion/` — Data collection modules
- `preprocessing/` — Text cleaning and normalization
- `models/` — Sentiment and intent analysis
- `visualization/` — Plots and dashboard

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Configure Reddit API credentials in `ingestion/reddit_config.py`.
2. Run the pipeline:
   ```bash
   python main.py
   ```
3. Outputs (CSV, plots) will be saved in the project directory.

## Example Outputs
- Example CSVs and plots will be generated after running the pipeline.

## Extending the Pipeline
- To add new platforms, create a new module in `ingestion/` and update the pipeline.

---

For detailed usage and examples, see the docstrings in each module.
