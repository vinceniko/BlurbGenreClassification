"""
main flow from preprocessing to model
"""

import load_p
import genres_p
import tokens_p

import pandas as pd


def get_preprocess_df(ds_path: str, ds_dir=load_p.DS_DIR) -> pd.DataFrame:
    """
    load the ds and preprocess it
    """
    df = load_p.get_df_flow(ds_path, ds_dir)
    df = genres_p.parse_genres_flow(df)

    # continue

    return df

if __name__ == "__main__":
    pass