import praw
import pandas as pd
from .reddit_config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
from typing import List, Dict

class RedditIngestor:
    def __init__(self, subreddit: str, limit: int = 100):
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        self.subreddit = subreddit
        self.limit = limit

    def fetch_posts(self) -> List[Dict]:
        posts = []
        for submission in self.reddit.subreddit(self.subreddit).hot(limit=self.limit):
            posts.append({
                'id': submission.id,
                'author': str(submission.author),
                'timestamp': submission.created_utc,
                'score': submission.score,
                'subreddit': submission.subreddit.display_name,
                'title': submission.title,
                'text': submission.selftext
            })
        return posts

    def fetch_comments(self, submission_id: str) -> List[Dict]:
        submission = self.reddit.submission(id=submission_id)
        submission.comments.replace_more(limit=0)
        comments = []
        for comment in submission.comments.list():
            comments.append({
                'id': comment.id,
                'author': str(comment.author),
                'timestamp': comment.created_utc,
                'score': comment.score,
                'subreddit': submission.subreddit.display_name,
                'text': comment.body
            })
        return comments

    def save_to_csv(self, data: List[Dict], filename: str):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

if __name__ == "__main__":
    ingestor = RedditIngestor(subreddit="python", limit=10)
    posts = ingestor.fetch_posts()
    ingestor.save_to_csv(posts, "reddit_posts.csv")
    # Fetch comments for the first post as an example
    if posts:
        comments = ingestor.fetch_comments(posts[0]['id'])
        ingestor.save_to_csv(comments, "reddit_comments.csv")
