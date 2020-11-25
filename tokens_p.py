"""
tokenizes the blurb bodies 
"""

from scipy.sparse import csr_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from typing import List, Tuple

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def preview_features(
        bodies: List[str],
        features: csr_matrix, term_names: List[str],
        doc_idx: int, num_features: int = 30):
    """
    previews the features and the body which they were taken from.

    params:
    features is a 2D numpy sparse matrix
    """
    doc_scores = features[doc_idx, :].todense()

    term_idxs_sorted = np.fliplr(doc_scores.argsort())

    for i in range(num_features):
        term_idx_inorder = term_idxs_sorted[0, i]

        term = term_names[term_idx_inorder]
        tfidf = doc_scores[0, term_idx_inorder]

        print('{0:20s} {1:f} '.format(term, tfidf))
    print()
    print(bodies[doc_idx])


def tokenize_flow(df: pd.DataFrame, **tf_params) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    tokenizes the blurb bodies in df and returns the sklearn vectorizer object and the feature matrix
    """
    if not 'stop_words' in tf_params:
        tf_params['stop_words'] = stopwords.words('english') + OUR_STOP_WORDS

    vectorizer = TfidfVectorizer(**tf_params)
    corpus = df['body']
    X = vectorizer.fit_transform(corpus)

    return vectorizer, X


OUR_STOP_WORDS = [
    'would'
]


if __name__ == "__main__":
    import load_p
    import genres_p

    import os

    ds_name: str = [f for f in os.listdir(load_p.DS_DIR) if 'train' in f][0]
    df = load_p.get_df_flow(ds_name)
    df = genres_p.parse_genres_flow(df)

    vectorizer, X = tokenize_flow(df)

    preview_features(df['body'], X, vectorizer.get_feature_names(), 113, 50)
