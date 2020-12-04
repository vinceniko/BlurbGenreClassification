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


remove_genres = [
    'Women’s Fiction',
    'Psychology',
    'Poetry',
    'Historical Fiction',
    'Humor',
    'Classics',
    'Travel',
    'Sports',
    'Parenting',
    'Games',
    'Gothic & Horror',
    'Pets',
    'Spiritual Fiction',
    'Military Fiction',
]

# TODO: create multiple such functions for each model variant
def extract_d0s_replace(
        df: pd.DataFrame, replace=['fiction', 'nonfiction']) -> pd.DataFrame:
    """
    keep only the d0 genres

    # no fiction-non
    Children’s Books          19499
    Mystery & Suspense         5796
    Graphic Novels & Manga     3793
    Religion & Philosophy      3775
    Teen & Young Adult         3510
    Literary Fiction           3098
    Arts & Entertainment       1983
    Romance                    1980
    Cooking                    1555
    History                    1529
    Biography & Memoir         1513
    Fantasy                    1232
    Popular Science            1184
    Self-Improvement           1140
    Science Fiction            1116
    Politics                    967
    Reference                   835
    Crafts, Home & Garden       716
    Health & Fitness            696
    Western Fiction             687
    Business                    650
    Women’s Fiction             545
    Psychology                  520
    Poetry                      504
    Historical Fiction          427
    Humor                       427
    Classics                    421
    Travel                      382
    Sports                      373
    Parenting                   306
    Games                       183
    Gothic & Horror             107
    Pets                         96
    Spiritual Fiction            92
    Military Fiction             42

    Children’s Books          0.316137
    Mystery & Suspense        0.093970
    Graphic Novels & Manga    0.061496
    Religion & Philosophy     0.061204
    Teen & Young Adult        0.056908
    Literary Fiction          0.050228
    Arts & Entertainment      0.032150
    Romance                   0.032102
    Cooking                   0.025211
    History                   0.024790
    Biography & Memoir        0.024530
    Fantasy                   0.019974
    Popular Science           0.019196
    Self-Improvement          0.018483
    Science Fiction           0.018094
    Politics                  0.015678
    Reference                 0.013538
    Crafts, Home & Garden     0.011608
    Health & Fitness          0.011284
    Western Fiction           0.011138
    Business                  0.010538
    Women’s Fiction           0.008836
    Psychology                0.008431
    Poetry                    0.008171
    Humor                     0.006923
    Historical Fiction        0.006923
    Classics                  0.006826
    Travel                    0.006193
    Sports                    0.006047
    Parenting                 0.004961
    Games                     0.002967
    Gothic & Horror           0.001735
    Pets                      0.001556
    Spiritual Fiction         0.001492
    Military Fiction          0.000681

    # no fiction-non and remove_genres
    Children’s Books          19499
    Mystery & Suspense         5796
    Graphic Novels & Manga     3793
    Religion & Philosophy      3775
    Teen & Young Adult         3510
    Literary Fiction           3098
    Arts & Entertainment       1983
    Romance                    1980
    Cooking                    1555
    History                    1529
    Biography & Memoir         1513
    Fantasy                    1232
    Popular Science            1184
    Self-Improvement           1140
    Science Fiction            1116
    Politics                    967
    Reference                   835
    Crafts, Home & Garden       716
    Health & Fitness            696
    Western Fiction             687
    Business                    650

    Children’s Books          0.340570
    Mystery & Suspense        0.101233
    Graphic Novels & Manga    0.066249
    Religion & Philosophy     0.065934
    Teen & Young Adult        0.061306
    Literary Fiction          0.054110
    Arts & Entertainment      0.034635
    Romance                   0.034583
    Cooking                   0.027160
    History                   0.026706
    Biography & Memoir        0.026426
    Fantasy                   0.021518
    Popular Science           0.020680
    Self-Improvement          0.019911
    Science Fiction           0.019492
    Politics                  0.016890
    Reference                 0.014584
    Crafts, Home & Garden     0.012506
    Health & Fitness          0.012156
    Western Fiction           0.011999
    Business                  0.011353
    """
    def _extract_d0(cell):
        d0 = cell['topics']['d0']
        out = d0
        if isinstance(d0, list):
            return None  # drop the items with multiple d0 genres
        elif d0.lower() in replace:
            d0_lower = d0.lower()
            if 'd1' in cell['topics']:
                d1 = cell['topics']['d1']
                if isinstance(d1, list):
                    return None
                out = d1
            else:
                # fiction-non
                if d0_lower in ['nonfiction', 'fiction']:
                    return None
        # after fiction-non
        # if out in remove_genres + ['Children’s Books']:
        if out in remove_genres:
            return None
        return out

    df_d0 = df['metadata'].map(_extract_d0)

    df['genre'] = df_d0
    # for converting into numerical form and extracting y
    df['genre'] = pd.Categorical(df['genre'])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def parse_genres_flow(df: pd.DataFrame, f_extract=extract_d0s_replace, *args, **kwargs) -> pd.DataFrame:
    """
    parse the genres and store them in the genre column of df
    pass in an extraction function to f_extract
    """
    df = f_extract(df, *args, **kwargs)

    return df


def count_genres(df: pd.DataFrame) -> pd.Series:
    genre_counts = df['genre'].value_counts()
    return genre_counts


def count_genres_perc(df: pd.DataFrame) -> pd.Series:
    genre_counts = df['genre'].value_counts()
    return genre_counts / genre_counts.sum()


def equalize_genre_size(
        df: pd.DataFrame, class_size: int = None) -> pd.DataFrame:
    if class_size is None:
        class_size = count_genres(df).min()
    df_eq = df.groupby('genre', as_index=False).nth(list(range(class_size)))
    df_eq.reset_index(drop=True, inplace=True)

    return df_eq


if __name__ == "__main__":
    import load_p
    import os

    df = load_p.get_df_flow()

    df = parse_genres_flow(df, extract_d0s_replace)
    print(count_genres(df))
    print(count_genres_perc(df))
    print(df.shape[0])
