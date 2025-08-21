import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df: pd.DataFrame, sentiment_col: str = 'sentiment'):
    plt.figure(figsize=(6,4))
    sns.countplot(x=sentiment_col, data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_intent_distribution(df: pd.DataFrame, intent_col: str = 'intent'):
    plt.figure(figsize=(6,4))
    sns.countplot(x=intent_col, data=df)
    plt.title('Intent Distribution')
    plt.xlabel('Intent')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage with dummy data
    data = {'sentiment': ['POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE'],
            'intent': ['food', 'travel', 'music', 'weather', 'food']}
    df = pd.DataFrame(data)
    plot_sentiment_distribution(df)
    plot_intent_distribution(df)
