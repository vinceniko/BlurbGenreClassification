from typing import Tuple
# external types
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def get_pca_features(X: csr_matrix) -> csr_matrix:
    # svd = TruncatedSVD(n_components=2000)
    svd = TruncatedSVD(n_components=100)
    # svd = TruncatedSVD(n_components=200)
    Z = svd.fit_transform(X)
    
    # ANALYSIS
    pov = np.cumsum(svd.explained_variance_ratio_)
    print(pov[:-10])

    return Z

if __name__ == "__main__":
    import load_p
    import genres_p
    import tokens_p

    import os

    df = load_p.get_df_flow()
    df = genres_p.parse_genres_flow(df)
    # min_df=10, 0.32191406
    # min_df=20, n_features=23540
    # can also change max_df (fraction)
    # without min_df, explained variance fell
    # setting max_df to max_df=1 / df['genre'].unique().size lowered pov = 0.47856286
    # using idf gives better results
    vectorizer, X = tokens_p.tokenize_flow(df, min_df=20, max_features=18000)
    tokens_p.preview_features(df['body'], X, vectorizer.get_feature_names(), 113, 20)

    Z = get_pca_features(X)

    print(Z)
