import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.svm as svm


def get_y(df: pd.DataFrame) -> np.array:
    y = df['genre'].cat.codes.values

    return y


# min_df=10
# 0.32191406
# min_df=15
# 0.41578408
# min_df=15
# 0.48123089
# min_df=20
# 0.49648864
# min_df=20, max_features=18000
# 0.48254273
# TODO: train test split on feature set and genre

# if __name__ == '__main__':

from sklearn.model_selection import train_test_split 

import load_p
import genres_p
import tokens_p
import feature_selection_p

import os

df = load_p.get_df_flow()
df = genres_p.equalize_genre_size(genres_p.parse_genres_flow(df, genres_p.extract_d0s_replace))
# min_df=10, 0.32191406
# min_df=20, n_features=23540
# can also change max_df (fraction)
# without min_df, explained variance fell
# setting max_df to max_df=1 / df['genre'].unique().size lowered pov = 0.47856286
# using idf gives better results
vectorizer, X = tokens_p.tokenize_flow(df, min_df=20, max_features=18000)
tokens_p.preview_features(df['body'], X, vectorizer.get_feature_names(), 113, 20)

Z = feature_selection_p.get_pca_features(X)
y = get_y(df)

Xtr, Xts, ytr, yts, Ztr, Zts = train_test_split(X, y, Z, random_state=0)

# genre removed 0.7767919519351684
svc = svm.SVC()
svc.fit(Ztr, ytr)
yhat = svc.predict(Zts)

svm_accuracy = np.mean(yhat == yts)
print(svm_accuracy) # n_cmp = 100, 0.8953729759653088, n_cmp = 200, 0.9050145315311159

# from joblib import dump, load
# dump(svc, 'svc_ncmp-100-remove_genres.joblib')
# svc_2 = load('svc_ncmp-100-remove_genres.joblib')

df['genre'].value_counts()
# Nonfiction            31911
# Fiction               30436
# Children’s Books      19499
# Teen & Young Adult     3510
# Poetry                  504
# Humor                   427
# Classics                421

from sklearn.naive_bayes import ComplementNB
clf = ComplementNB()
clf.fit(Xtr, ytr)
yhat = clf.predict(Xts)

clf_accuracy = np.mean(yhat == yts)
print(clf_accuracy) # 0.852378096600083

# for genre in misclassified.index:
#     row = misclassified.loc[genre]
#     acc = row[genre]
#     if acc < 0.75:
#         print(genre, acc, "misclassified as: ", sorted([(misclassified.columns[i], acc_2) for i, acc_2 in enumerate(row) if acc_2 > 0], key=lambda x: x[1])[::-1])
#         print()
# Biography & Memoir 0.47 misclassified as:  [('Biography & Memoir', 0.47), ('Children’s Books', 0.12), ('Literary Fiction', 0.1), ('History', 0.08), ('Politics', 0.05), ('Mystery & Suspense', 0.05), ('Religion & Philosophy', 0.03), ('Teen & Young Adult', 0.02), ('Self-Improvement', 0.02), ('Romance', 0.02), ('Business', 0.02), ('Arts & Entertainment', 0.02), ('Popular Science', 0.01), ('Health & Fitness', 0.01)]
# Business 0.7 misclassified as:  [('Business', 0.7), ('Self-Improvement', 0.14), ('Politics', 0.08), ('History', 0.03), ('Biography & Memoir', 0.02), ('Religion & Philosophy', 0.01), ('Popular Science', 0.01), ('Health & Fitness', 0.01), ('Crafts, Home & Garden', 0.01), ('Arts & Entertainment', 0.01)]
# Fantasy 0.62 misclassified as:  [('Fantasy', 0.62), ('Children’s Books', 0.11), ('Teen & Young Adult', 0.08), ('Graphic Novels & Manga', 0.06), ('Mystery & Suspense', 0.05), ('Literary Fiction', 0.04), ('Science Fiction', 0.02), ('Romance', 0.01), ('Religion & Philosophy', 0.01)]
# Graphic Novels & Manga 0.72 misclassified as:  [('Graphic Novels & Manga', 0.72), ('Children’s Books', 0.14), ('Mystery & Suspense', 0.04), ('Teen & Young Adult', 0.03), ('Literary Fiction', 0.03), ('Science Fiction', 0.01), ('Fantasy', 0.01), ('Arts & Entertainment', 0.01)]
# History 0.64 misclassified as:  [('History', 0.64), ('Politics', 0.07), ('Biography & Memoir', 0.07), ('Religion & Philosophy', 0.05), ('Children’s Books', 0.05), ('Mystery & Suspense', 0.04), ('Literary Fiction', 0.04), ('Arts & Entertainment', 0.03), ('Popular Science', 0.02), ('Science Fiction', 0.01)]
# Literary Fiction 0.72 misclassified as:  [('Literary Fiction', 0.72), ('Children’s Books', 0.08), ('Teen & Young Adult', 0.06), ('Mystery & Suspense', 0.04), ('Biography & Memoir', 0.03), ('History', 0.02), ('Romance', 0.01), ('Religion & Philosophy', 0.01), ('Graphic Novels & Manga', 0.01)]
# Politics 0.6 misclassified as:  [('Politics', 0.6), ('History', 0.13), ('Religion & Philosophy', 0.05), ('Children’s Books', 0.05), ('Biography & Memoir', 0.04), ('Popular Science', 0.03), ('Mystery & Suspense', 0.03), ('Self-Improvement', 0.02), ('Literary Fiction', 0.01), ('Health & Fitness', 0.01), ('Business', 0.01)]
# Popular Science 0.68 misclassified as:  [('Popular Science', 0.68), ('Children’s Books', 0.1), ('Religion & Philosophy', 0.06), ('History', 0.03), ('Science Fiction', 0.02), ('Politics', 0.02), ('Literary Fiction', 0.02), ('Health & Fitness', 0.02), ('Arts & Entertainment', 0.02), ('Reference', 0.01), ('Cooking', 0.01), ('Biography & Memoir', 0.01)]
# Science Fiction 0.65 misclassified as:  [('Science Fiction', 0.65), ('Children’s Books', 0.08), ('Graphic Novels & Manga', 0.07), ('Mystery & Suspense', 0.05), ('Teen & Young Adult', 0.04), ('Literary Fiction', 0.04), ('Fantasy', 0.03), ('History', 0.02), ('Popular Science', 0.01)]
# Self-Improvement 0.71 misclassified as:  [('Self-Improvement', 0.71), ('Children’s Books', 0.08), ('Religion & Philosophy', 0.07), ('Business', 0.07), ('Health & Fitness', 0.02), ('Biography & Memoir', 0.02), ('Reference', 0.01), ('Popular Science', 0.01), ('Arts & Entertainment', 0.01)]
# Teen & Young Adult 0.55 misclassified as:  [('Teen & Young Adult', 0.55), ('Children’s Books', 0.24), ('Literary Fiction', 0.05), ('Mystery & Suspense', 0.04), ('Graphic Novels & Manga', 0.03), ('Romance', 0.02), ('Fantasy', 0.02), ('Self-Improvement', 0.01), ('Science Fiction', 0.01), ('Religion & Philosophy', 0.01), ('Biography & Memoir', 0.01)]

cutoff = 0.75
print(f"over {cutoff}")
for genre in misclassified.index:
    row = misclassified.loc[genre]
    acc = row[genre]
    if acc >= cutoff:
        print(genre, acc, "misclassified as: ", sorted([(misclassified.columns[i], acc_2) for i, acc_2 in enumerate(row) if acc_2 > 0], key=lambda x: x[1])[::-1])
        print()

print(f"under {cutoff}")
for genre in misclassified.index:
    row = misclassified.loc[genre]
    acc = row[genre]
    if acc < cutoff:
        print(genre, acc, "misclassified as: ", sorted([(misclassified.columns[i], acc_2) for i, acc_2 in enumerate(row) if acc_2 > 0], key=lambda x: x[1])[::-1])
        print()

list(enumerate(df['genre'].cat.categories))
# [(0, 'Children’s Books'),
#  (1, 'Classics'),
#  (2, 'Fiction'),
#  (3, 'Humor'),
#  (4, 'Nonfiction'),
#  (5, 'Poetry'),
#  (6, 'Teen & Young Adult')]

list(enumerate(df['genre'].cat.categories))

list(enumerate(df['genre'].cat.categories))
# [(0, 'Arts & Entertainment'),
#  (1, 'Biography & Memoir'),
#  (2, 'Business'),
#  (3, 'Children’s Books'),
#  (4, 'Classics'),
#  (5, 'Cooking'),
#  (6, 'Crafts, Home & Garden'),
#  (7, 'Fantasy'),
#  (8, 'Fiction'),
#  (9, 'Games'),
#  (10, 'Gothic & Horror'),
#  (11, 'Graphic Novels & Manga'),
#  (12, 'Health & Fitness'),
#  (13, 'Historical Fiction'),
#  (14, 'History'),
#  (15, 'Humor'),
#  (16, 'Literary Fiction'),
#  (17, 'Military Fiction'),
#  (18, 'Mystery & Suspense'),
#  (19, 'Nonfiction'),
#  (20, 'Parenting'),
#  (21, 'Pets'),
#  (22, 'Poetry'),
#  (23, 'Politics'),
#  (24, 'Popular Science'),
#  (25, 'Psychology'),
#  (26, 'Reference'),
#  (27, 'Religion & Philosophy'),
#  (28, 'Romance'),
#  (29, 'Science Fiction'),
#  (30, 'Self-Improvement'),
#  (31, 'Spiritual Fiction'),
#  (32, 'Sports'),
#  (33, 'Teen & Young Adult'),
#  (34, 'Travel'),
#  (35, 'Western Fiction'),
#  (36, 'Women’s Fiction')]