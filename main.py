"""
main flow from preprocessing to model
"""

from typing import Tuple

import load_p
import genres_p
import tokens_p

import pandas as pd


def get_preprocess_df(ds_path: str, ds_dir=load_p.DS_DIR) -> pd.DataFrame:
    """
    load the ds and preprocesses it
    """
    df = load_p.get_df_flow(ds_path, ds_dir)
    df = genres_p.parse_genres_flow(df)

    return df


if __name__ == "__main__":
    import os

    ds_name: str = [f for f in os.listdir(load_p.DS_DIR) if 'train' in f][0]
    df = get_preprocess_df(ds_name)

    vectorizer, X = tokens_p.tokenize_flow(df)

    # continue
    # feature reduction, i.e. PCA