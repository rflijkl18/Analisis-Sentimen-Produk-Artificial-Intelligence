from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),     # Unigram + Bigram
        sublinear_tf=True,
        max_features=10000      # Untuk performa tinggi
    )),
    ('clf', MultinomialNB())
])
