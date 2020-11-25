import pandas as pd


def extract_d0s(df: pd.DataFrame) -> pd.DataFrame:
    """
    keep only the d0 genres
    """
    def _extract_d0(cell):
        cell = cell['topics']['d0']
        if isinstance(cell, list):
            return None
        return cell

    df_d0 = df['metadata'].map(_extract_d0)

    df['genre'] = df_d0

    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def parse_genres_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    parse the genres and store them in the genre column of df
    """
    df = extract_d0s(df)

    return df


if __name__ == "__main__":
    import load_p
    import os

    ds_name: str = [f for f in os.listdir(load_p.DS_DIR) if 'train' in f][0]
    df = load_p.get_df_flow(ds_name)

    df = parse_genres_flow(df)
    print(df.head())
