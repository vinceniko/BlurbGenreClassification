### preprocessing ###

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

import analysis_p
import load_p
import genres_p
import tokens_p
import feature_selection_p

import os

def get_y(df: pd.DataFrame) -> np.array:
    y = df['genre'].cat.codes.values

    return y

df = load_p.get_df_flow()
# df = genres_p.balance_genre_size(genres_p.parse_genres_flow(df, genres_p.extract_d0s_replace))
df = genres_p.parse_genres_flow(df, genres_p.extract_d0s_replace)

vectorizer, X = tokens_p.tokenize_flow(df, min_df=20, max_features=18000)
print(X.shape)
# confirm vectorizer worked
tokens_p.preview_features(df['body'], X, vectorizer.get_feature_names(), 113, 20)

Z = feature_selection_p.get_pca_features(X, n_components=100)
print(Z.shape)
y = get_y(df)

Xtr, Xts, ytr, yts, Ztr, Zts = train_test_split(X, y, Z, random_state=0)

### ComplementNB ###

from sklearn.naive_bayes import ComplementNB
clf = ComplementNB()
clf.fit(Xtr, ytr)
yhat_clf = clf.predict(Xts)

acc_clf = np.mean(yhat_clf == yts)
print(acc_clf)

### SVM ###

import sklearn.svm as svm
svc = svm.SVC()
svc.fit(Ztr, ytr)
yhat = svc.predict(Zts)

svm_accuracy = np.mean(yhat == yts)
print(svm_accuracy)

import sklearn.metrics as metrics
print(metrics.classification_report(yts, yhat, labels=range(df['genre'].cat.categories.values.size), target_names=df['genre'].cat.categories.values))
analysis_p.misclassified_analysis(yhat, yts, df, "svm_d1_unbalanced")

### NeuralNetwork ###

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D,Embedding

import tensorflow.keras.backend as K
K.clear_session()

batch_size = 500
n_epochs = 100
filters = 100
kernels = 3
train = Ztr
test = Zts

train = Ztr
test = Zts

num_classes = df['genre'].value_counts().index.size
model = Sequential()
model.add(Dense(num_classes*500, input_shape=train.shape[-1:], activation = 'relu'))
model.add(Dense(num_classes, activation='softmax'))
from tensorflow.keras import optimizers
opt = optimizers.Adam(lr=0.0001)
hist = model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
print(model.summary())

hist = model.fit(train, ytr, batch_size=batch_size,
              epochs=100, validation_data=(test, yts),
              shuffle=True)

tr_accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']

plt.plot(tr_accuracy)
plt.plot(val_accuracy)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('accuarcy')
plt.legend(['training accuracy', 'validation accuracy'])

yhat = model.predict_classes(Zts)
np.unique(yhat)

analysis_p.misclassified_analysis(yhat, yts, df, 'nn_equal_classes_large')

df['genre'].cat.categories

print(metrics.classification_report(yts, yhat, labels=range(num_classes), target_names=df['genre'].cat.categories.values))
