import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import os

OUTPUT_DIR = "outputs/"


def get_accuracy(yhat, yts):
    acc = np.mean(yhat == yts)
    print("accuracy:", acc)  # 0.852378096600083

    return acc


def get_misclassified(yhat, yts, df, output_name):
    # conf matrix
    fmt = np.vectorize(lambda x: round(float(x), 2))
    misclassified = pd.DataFrame(
        fmt(
            confusion_matrix(yts, yhat) / np.sum(
                confusion_matrix(yts, yhat),
                axis=1)[:, None]),
        index=df['genre'].cat.categories, columns=df['genre'].cat.categories)
    misclassified.to_csv(
        os.path.join(
            OUTPUT_DIR, "misclassified/{:s}_data.csv".format(output_name)))
    print(misclassified)

    return misclassified


def misclassified_plot(
        misclassified, output_name, cmap=sns.color_palette(
            "rocket_r", as_cmap=True)):
    # heatmap
    heatmap = sns.heatmap(misclassified, cmap=cmap, xticklabels=True)
    heatmap.get_figure().savefig(
        os.path.join(
            OUTPUT_DIR, "misclassified/{:s}_heatmap.png".format(output_name)),
        bbox_inches="tight")


def get_sorted_misclassified(misclassified, accuracy):
    cutoff = accuracy
    s = ""

    def write_cutoff(cutoff, greater):
        nonlocal s
        cutoff = round(cutoff, 2)
        for i, genre in enumerate(misclassified.index):
            row = misclassified.loc[genre]
            acc = row[genre]

            def wr():
                nonlocal s
                s += f"Genre: {genre}, Accuracy: {acc}, Classified as: {sorted([(misclassified.columns[i], acc_2) for i, acc_2 in enumerate(row) if acc_2 > 0], key=lambda x: x[1])[::-1]}\n\n"
            if greater:
                if i == 0:
                    s += f"Over: {cutoff}\n\n"
                if acc >= cutoff:
                    wr()
            else:
                if i == 0:
                    s += f"Under: {cutoff}\n\n"
                if acc < cutoff:
                    wr()

    write_cutoff(cutoff, True)
    write_cutoff(cutoff, False)

    return s


def misclassified_analysis(yhat, yts, df, output_name):
    misclassified = get_misclassified(yhat, yts, df, output_name)

    misclassified_plot(misclassified, output_name)

    accuracy = get_accuracy(yhat, yts)
    sorted_misclassified = get_sorted_misclassified(misclassified, accuracy)
    with open(os.path.join(OUTPUT_DIR, f"misclassified/{output_name}.txt"), "w") as f:
        print(sorted_misclassified, file=f)
