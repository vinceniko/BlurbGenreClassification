"""
loads the df
"""

from typing import List
from collections import OrderedDict

import xml.etree.ElementTree as ET
import xmltodict

import pandas as pd

import os


Books = List[OrderedDict]

DS_DIR = "blurbgenrecollectionen/datasets"


def load_ds(f_path: str) -> Books:
    """
    loads the XML dataset from the file path and returns a List of Books stored in OrderedDicts
    """
    with open(f_path, "r") as f:
        f_str = f.read()

    f_str = "<data>\n" + f_str
    f_str = f_str + "</data>\n"
    f_str = f_str.replace("&", "&amp;")
    xml_dct = xmltodict.parse(f_str)

    return xml_dct['data']['book']


def to_df(books: Books) -> pd.DataFrame:
    """
    converts Books list of dictionaries into a dataframe
    """
    return pd.DataFrame.from_records(books)


def get_df(ds_path: str, ds_dir = DS_DIR) -> pd.DataFrame:
    """
    load the ds and preprocess it
    """
    books: Books = load_ds(os.path.join(DS_DIR, ds_path))
    df = to_df(books)
    
    return df


def get_df_keyword(dir_path: str, keyword: str='train'):
    """
    get df from dir_path using a keyword
    """
    ds_name: str = [f for f in os.listdir(dir_path) if keyword in f][0]
    print('loading:', ds_name)
    df = get_df(ds_name)

    return df


def get_entire_ds_as_df(ds_dir = DS_DIR, f_name_keywords=['train', 'dev', 'test']):
    """
    used to concatenate all sets into one
    """
    df = pd.DataFrame()
    for kind in f_name_keywords:
        df_temp = get_df_keyword(ds_dir, kind)
        df_temp.set_index('title', inplace=True)
        df = pd.concat([df, df_temp], ignore_index=False)
    df.reset_index()

    return df


def get_df_flow(ds_dir = DS_DIR) -> pd.DataFrame:
    return get_entire_ds_as_df(ds_dir)


if __name__ == "__main__":
    df = get_df_flow()
    print(df.shape)