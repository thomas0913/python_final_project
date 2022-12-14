from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks

train_url = "./excel/train.csv"
test_url = "./excel/test.csv"
sample_submission_url = "./excel/sample_submission.csv"

# ===================
#    資 料 預 處 理
# ===================
# 建立 DataFrame
df_train = pd.read_csv(train_url)
df_test = pd.read_csv(test_url)
df_sample_submission = pd.read_csv(sample_submission_url)

# 評估缺失值
print((df_train.isnull().sum()/len(df_train)), "\n")
print((df_test.isnull().sum()/len(df_test)), "\n")
print((df_sample_submission.isnull().sum()/len(df_sample_submission)), "\n")

# ===================
#     資 料 分 割
# ===================
train_data = []
test_data = []
train_target = []
test_target = df_sample_submission.values[:, 1]

# ===================
#     資 料 準 備
# ===================

# ===================
#     模 型 訓 練
# ===================