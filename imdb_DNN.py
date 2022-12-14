import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
from tensorflow.keras.datasets import imdb

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
# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000

model = ks.Sequential()
model.add(ks.layers.Embedding(vocab_size, 16))
model.add(ks.layers.GlobalAveragePooling1D())
model.add(ks.layers.Dense(16, activation='relu'))
model.add(ks.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_target[:10000]
partial_y_train = train_target[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

"""
Loss function scoring
"""
results = model.evaluate(test_data,  test_target, verbose=2)

print(results)
