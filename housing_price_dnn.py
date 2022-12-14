import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.datasets.boston_housing as boston_ds
from sklearn.preprocessing import StandardScaler

# Data Preprocessing
(train_data, train_target), (test_data, test_target) = boston_ds.load_data(path='boston_housing.npz', test_split=0.25, seed=13)
std_train_data = StandardScaler().fit_transform(train_data)
std_test_data = StandardScaler().fit_transform(test_data)
std_train_target = StandardScaler().fit_transform(train_target.reshape(-1, 1))
std_test_target = StandardScaler().fit_transform(test_target.reshape(-1, 1))

# Model Definition
def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(std_train_data.shape[1], )))
    for i in range(15):
        model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Building Model
num_epochs = 10
num_batch = 10
model = build_model()
model.fit(std_train_data, std_train_target, epochs=num_epochs, batch_size=num_batch)
train_pred = model.predict(std_train_data)
test_pred = model.predict(std_test_data)

#Training and Evaluating
(test_mse_score, test_mae_score) = model.evaluate(std_test_data, std_test_target)
print('test mse score :%.3f, mae score:%.3f' %(test_mse_score, test_mae_score))

print("\n")