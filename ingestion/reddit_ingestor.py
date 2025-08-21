import praw
import pandas as pd
from .reddit_config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
from typing import List, Optional, Any

class RedditIngestor:
    def __init__(self, subreddit: str, limit: int = 100):
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        self.subreddit = subreddit
        self.limit = limit


    def fetch_top_posts(self, limit: Optional[int] = None) -> List[dict[str, Any]]:
        """
        Fetch top posts by score (engagement). Uses Reddit's 'top' listing.
        """
        posts = []
        for submission in self.reddit.subreddit(self.subreddit).top(limit=limit or self.limit):
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

    def fetch_keyword_posts(self, keywords: List[str], limit: Optional[int] = None, min_score: int = 0) -> List[dict[str, Any]]:
        """
        Fetch posts matching any of the provided keywords, with optional engagement threshold.
        """
        posts = []
        query = ' OR '.join([f'"{kw}"' for kw in keywords])
        for submission in self.reddit.subreddit(self.subreddit).search(query, sort='relevance', limit=limit or self.limit):
            if submission.score >= min_score:
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

    def fetch_posts(self, mode: str = 'top', keywords: Optional[List[str]] = None, limit: Optional[int] = None, min_score: int = 0) -> List[dict[str, Any]]:
        """
        Flexible ingestion: mode can be 'top', 'keyword', or 'both'.
        - 'top': fetch top posts by engagement
        - 'keyword': fetch posts matching keywords (multi-keyword, OR logic)
        - 'both': fetch both and return combined (deduplicated by post id)
        """
        limit = limit or self.limit
        if mode == 'top':
            return self.fetch_top_posts(limit=limit)
        elif mode == 'keyword' and keywords:
            return self.fetch_keyword_posts(keywords, limit=limit, min_score=min_score)
        elif mode == 'both' and keywords:
            top_posts = self.fetch_top_posts(limit=limit)
            keyword_posts = self.fetch_keyword_posts(keywords, limit=limit, min_score=min_score)
            # Deduplicate by post id
            all_posts = {p['id']: p for p in top_posts + keyword_posts}
            return list(all_posts.values())
        else:
            raise ValueError("Invalid mode or missing keywords for keyword search.")

    def fetch_comments(self, submission_id: str) -> List[dict[str, Any]]:
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

    def save_to_csv(self, data: List[dict], filename: str):
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
