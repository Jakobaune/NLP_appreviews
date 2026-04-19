# Sentiment Analysis of AI Assistant App Reviews

Project for TDT4310 – Intelligent Text Analytics and Language Understanding, NTNU.

Classifies Google Play reviews for ChatGPT, Gemini, and Claude into three sentiment categories (Positive, Neutral, Negative) and compares five approaches: lexicon-based baselines, Naïve Bayes, feature engineering, and fine-tuned RoBERTa.

---

## Results summary

| System | Macro F1 |
|---|---|
| SentiWordNet | 0.442 |
| VADER | 0.475 |
| Naïve Bayes + TF-IDF | 0.574 |
| Naïve Bayes + Lemmas | 0.575 |
| Naïve Bayes + TF-IDF + POS | 0.572 |
| Naïve Bayes + TF-IDF + LDA | 0.571 |
| RoBERTa (balanced) | **0.661** |
| RoBERTa (full + class-weighted) | 0.659 |

---

## Repository structure

```
utils.py                        # Shared helpers (classify_sentiment, normalize_text, balance_by_min_class)
data/
  reviewScraper.ipynb           # Scrapes reviews, deduplicates, creates stratified 80/20 train/test splits
eda/
  eda.ipynb                     # Exploratory data analysis and class distribution
  spacy_analysis.ipynb          # Lemmatization and POS tagging analysis
  topic_modeling_ner.ipynb      # LDA topic modeling and named entity recognition
NLP/
  lexicon_baselines.ipynb       # VADER and SentiWordNet lexicon baselines
  naive_bayes.ipynb             # Naïve Bayes variants (TF-IDF, lemmas, POS, LDA)
  RoBERTa_balanced.ipynb        # RoBERTa fine-tuned on downsampled balanced training set
  RoBERTa_Full_Dataset.ipynb    # RoBERTa fine-tuned on imbalanced training set with class-weighted loss
```

---

## Data

Reviews were scraped from Google Play (English, US region) for three apps:
- ChatGPT – 100,000 reviews
- Gemini – 100,000 reviews
- Claude – 33,329 reviews

Sentiment labels are derived from star ratings: ≥ 4 → Positive, 3 → Neutral, ≤ 2 → Negative.

The raw and processed CSV files are not included in this repository due to size. Run `data/reviewScraper.ipynb` top-to-bottom to reproduce them.

---

## Setup

```bash
pip install pandas scikit-learn nltk spacy gensim transformers torch google-play-scraper tqdm
python -m spacy download en_core_web_sm
```

Python 3.11. RoBERTa training was run on Apple MPS (Mac GPU). GPU or CUDA is strongly recommended — training takes ~30 minutes per run on MPS.

---

## Running order

1. `data/reviewScraper.ipynb` — scrape, clean, and split the data
2. `eda/eda.ipynb` — explore class distributions
3. `eda/` notebooks — optional linguistic analysis
4. `NLP/lexicon_baselines.ipynb` — VADER and SentiWordNet
5. `NLP/naive_bayes.ipynb` — Naïve Bayes variants
6. `NLP/RoBERTa_balanced.ipynb` — RoBERTa on balanced data
7. `NLP/RoBERTa_Full_Dataset.ipynb` — RoBERTa with class-weighted loss
