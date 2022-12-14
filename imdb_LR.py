import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
from tensorflow.keras.datasets import imdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

# 参数 num_words=10000 保留了训练数据中最常出现的 10,000 个单词。为了保持数据规模的可管理性，低频词将被丢弃。
(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=10000)

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

train_data = ks.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = ks.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

"""
Logistic Regression module
"""
LR = LogisticRegression(penalty='l2')
LR.fit(train_data, train_target.flatten()) # 利用已知數據集來求出最佳模型曲線
train_pred = LR.predict(train_data)
train_pred_proba = LR.predict_proba(train_data)
test_pred = LR.predict(test_data)
test_pred_proba = LR.predict_proba(test_data)

"""
Loss function scoring
"""
print('the score is :%3f' %(LR.score(train_data, train_target)))
print('the score is :%3f' %(LR.score(test_data, test_target)))
print('the accuracy score is :%3f' %(accuracy_score(train_target, train_pred)))
print('the accuracy score is :%3f' %(accuracy_score(test_target, test_pred)))
print('performance :%.3f' %(log_loss(train_target, train_pred_proba)))
print('performance :%.3f' %(log_loss(test_target, test_pred_proba)))

print("")