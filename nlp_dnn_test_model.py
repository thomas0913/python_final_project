from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from lib.text_removal import TextRemoval
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from text_preprocessing import TextPreprocessing
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy 
import tensorflow.keras.preprocessing as data_preprocessor
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
import re
import contractions
import nltk

train_url = "./excel/train.csv"
test_url = "./excel/test.csv"
sample_submission_url = "./excel/sample_submission.csv"

# ===================
#     資 料 載 入
# ===================
TextPreprocessor = TextPreprocessing()
df_train, df_test = TextPreprocessor.load_datasets(train_url, test_url)

# ===================
#     資 料 準 備
# ===================
# Zero Padding
MAX_SQUENCE_LENGTH = max([len(seq) for seq in df_train["tokenized"]])
data = data_preprocessor.sequence.pad_sequences(df_train["tokenized"], padding='post')
target = df_train['target'].values[:]

# 資料分割
train_data, valid_data, train_target, valid_target = train_test_split(data, target, test_size=0.25, random_state=20)
print(train_data.shape)
print(valid_data.shape)
print(train_target.shape)
print(valid_target.shape)

# ===================
#     模 型 訓 練
# ===================

def build_model():
    model = ks.models.Sequential()
    model.add(layers.Embedding(10000, 256))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=True), metrics=[BinaryAccuracy(threshold=0.0, name='accuracy')])
    return model

model = build_model()
history = model.fit(train_data, train_target, epochs=40, batch_size=512, validation_data=(valid_data, valid_target), verbose=1)
test_mse_score, test_mae_score = model.evaluate(valid_data, valid_target)
print('test mse score :%.3f, mae score:%.3f' %(test_mse_score, test_mae_score))
predict = model.predict(train_data)
print(predict)

"""
LR = LogisticRegression(penalty='l2', C=200)
LR.fit(train_data, train_target)
train_pred = LR.predict(train_data)
train_pred_proba = LR.predict_proba(train_data)
valid_pred = LR.predict(valid_data)
valid_pred_proba = LR.predict_proba(valid_data)

# ===================
#     模 型 評 估
# ===================
print('Train: the score is %3f' %(LR.score(train_data, train_target)))
print('Test : the score is %3f' %(LR.score(valid_data, valid_target)))
print('Train: performance %.3f' %(log_loss(train_target, train_pred_proba)))
print('Test : performance %.3f' %(log_loss(valid_target, valid_pred_proba)))
"""