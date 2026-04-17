"""Shared helpers used across notebooks.

Notebooks live in subfolders (eda/, NLP/, data/). Import pattern:

    import sys
    sys.path.insert(0, '..')
    from utils import classify_sentiment, normalize_text, balance_by_min_class
"""
import re

import pandas as pd


def classify_sentiment(score):
    """Map a 1-5 star rating to Positive / Neutral / Negative."""
    if score >= 4:
        return 'Positive'
    if score == 3:
        return 'Neutral'
    return 'Negative'


def normalize_text(s):
    """Strip newlines and collapse whitespace. Safe on None/NaN."""
    s = '' if s is None else str(s)
    s = s.replace('\n', ' ').replace('\r', ' ').strip()
    return re.sub(r'\s+', ' ', s)


def balance_by_min_class(df, label_col='Sentiment', random_state=42):
    """Downsample every class to the size of the smallest, then shuffle."""
    min_size = df[label_col].value_counts().min()
    balanced = (
        df.groupby(label_col, group_keys=False)
          .sample(n=min_size, random_state=random_state)
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )
    return balanced
