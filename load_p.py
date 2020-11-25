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


def get_df_flow(ds_path: str, ds_dir = DS_DIR) -> pd.DataFrame:
    """
    load the ds and preprocess it
    """
    books: Books = load_ds(os.path.join(DS_DIR, ds_path))
    df = to_df(books)
    
    return df


if __name__ == "__main__":
    ds_path_lst = os.listdir(DS_DIR)

    for ds_name in ds_path_lst:
        print(get_df_flow(ds_name))
