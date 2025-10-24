## Sentiment-Based Laptop Recommendation System
## Project Overview

This project is an NLP-based recommendation system that analyzes user reviews from various e-commerce platforms to recommend the top five laptops with the most positive feedback.

By applying sentiment analysis, it identifies the emotional tone of customer comments and ranks products accordingly.

## Data Collection

Sources: Trendyol, N11, Vatan Bilgisayar, Amazon, and Google Shopping

Method: Web scraping with Python Selenium

Dataset: 22,562 reviews and 710 laptops

## Data Preprocessing

Text cleaning: removed capitalization, punctuation, emojis, and unnecessary spaces

Removed Turkish stop words

Tokenization with BERT Tokenizer (Subword Tokenization) trained on Turkish data

## Semi-Supervised Learning

First 1,000 reviews manually labeled as Positive, Neutral, Negative

Remaining reviews labeled using Pseudo-Labeling and Iterative Pseudo-Labeling

Improved performance by adding confident predictions into the training set iteratively

## Models & Embeddings

Tested Embeddings: TF-IDF, Word2Vec, GloVe, FastText

Algorithms: Naive Bayes, SVM, Logistic Regression, Random Forest, LSTM

| Embedding | Best Model                 | F1-Score | Notes                              |
| --------- | -------------------------- | -------- | ---------------------------------- |
| TF-IDF    | Logistic Regression        | ~0.90    | Best performance overall           |
| Word2Vec  | LSTM                       | ~0.90    | Captured contextual relations well |
| GloVe     | Logistic Regression        | ~0.74    | Poor for imbalanced data           |
| FastText  | Logistic Regression / LSTM | ~0.87    | Robust and balanced                |


## Handling Imbalance

Used SMOTE (Synthetic Minority Oversampling Technique) to balance classes and reduce overfitting.

## Recommendation Logic

Final model: Naive Bayes + TF-IDF (after SMOTE)
Score formula:

Score =(Positive Reviews / Total Reviews)× log(Total Reviews)

Top 5 laptops are recommended based on this score within a selected price range.
