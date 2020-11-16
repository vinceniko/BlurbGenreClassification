"""
tokenizes the blurb bodies 
"""

from string import punctuation
import pandas as pd
import unicodedata
import sys
from typing import List, Set
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def tokenize(df: pd.DataFrame, src_col: str = "body", dst_col: str = "body_tokens") -> pd.DataFrame:
    """
    tokenizes (also removes stop-words and punctuation) the body text of src_col and stores it in dst_col of df
    """
    def _tokenize(body: str) -> List[str]:
        stop = stopwords.words('english') + list(punctuation)
        return [i for i in word_tokenize(body.lower()) if i not in stop]

    df[dst_col] = df[src_col].map(_tokenize)

    return df


def remove_utf_punctuation(
        df: pd.DataFrame, src_col: str = 'body_tokens',
        dst_col: str = 'body_tokens_nopunc') -> pd.DataFrame:
    """
    removes unicode punctuation not removed by tokenize
    """

    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)).startswith('P'))

    def _remove_utf_punctuation(tokens: List[str]) -> List[str]:
        new_tokens = []

        for token in tokens:
            punc_removed = token.translate(tbl)
            if punc_removed != '' or punc_removed is not None:
                new_tokens.append(punc_removed)

        return new_tokens

    df[dst_col] = df[src_col].map(_remove_utf_punctuation)

    return df


def get_unique_tokens(df: pd.DataFrame, src_col: str = "body_tokens_nopunc") -> Set:
    """
    returns the unique tokens in the token lists of src_col
    """
    words = set()

    def _add_words(tokens: List[str]) -> None:
        for token in tokens:
            words.add(token)

    df[src_col].map(_add_words)

    return words


def tokenize_flow(df: pd.DataFrame) -> pd.DataFrame:
    df = tokenize(df)
    df = remove_utf_punctuation(df)

    return df


if __name__ == "__main__":
    import load_p
    import genres_p
    import os

    ds_name: str = [f for f in os.listdir(load_p.DS_DIR) if 'train' in f][0]
    df = load_p.get_df_flow(ds_name)
    df = genres_p.parse_genres_flow(df)

    df = tokenize_flow(df)
    print(df.head())

    words = get_unique_tokens(df)
    print("len words:", len(words))
