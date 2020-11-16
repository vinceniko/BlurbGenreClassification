from typing import List, OrderedDict

import xml.etree.ElementTree as ET
import xmltodict

import pandas as pd

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


def preprocess_d0s(df: pd.DataFrame) -> pd.DataFrame:
    def extract_d0(cell):
        cell = cell['topics']['d0']
        if isinstance(cell, list):
            return None
        return cell

    df_d0 = df['metadata'].map(extract_d0)

    df['d0'] = df_d0
    df.dropna(inplace=True)

    return df


def preprocess(books: Books) -> pd.DataFrame:
    df = pd.DataFrame.from_records(books)

    df = preprocess_d0s(df)

    return df


if __name__ == "__main__":
    import os

    ds_path_lst = os.listdir(DS_DIR)

    for ds_path in ds_path_lst:
        books: Books = load_ds(os.path.join(DS_DIR, ds_path))
        print(preprocess(books).head())
