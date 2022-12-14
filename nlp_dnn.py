from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
import re
import contractions

train_url = "./excel/train.csv"
test_url = "./excel/test.csv"
sample_submission_url = "./excel/sample_submission.csv"

# ===================
#     資 料 匯 入
# ===================
# 建立 DataFrame
df_train = pd.read_csv(train_url)
df_test = pd.read_csv(test_url)
df_sample_submission = pd.read_csv(sample_submission_url)

# 評估缺失值
print((df_train.isnull().sum()/len(df_train)), "\n")
print((df_test.isnull().sum()/len(df_test)), "\n")
print((df_sample_submission.isnull().sum()/len(df_sample_submission)), "\n")

# 檢視部分資料內容
print(df_train.head(3))
print(df_test.head(3))

df_train = df_train.loc[:, ['id', 'text', 'target']]
df_test = df_test.loc[:, ['id', 'text']]

# ===================
#     資 料 分 析
# ===================

# ===================
#     資 料 過 濾
# ===================
# 文本小寫化
df_train["text_clean"] = df_train["text"].apply(lambda x: x.lower())
# 文本縮詞展開
df_train["text_clean"] = df_train["text_clean"].apply(lambda x: contractions.fix(x))
# 文本雜訊過濾

# ===================
#   資 料 前 處 理
# ===================
# 過濾非內文資料


print(df_train.head(3))

# 文本斷詞

# 文本標記

# ===================
#     資 料 準 備
# ===================
train_data = []
test_data = []
train_target = []
test_target = df_sample_submission.values[:, 1]

# ===================
#     模 型 訓 練
# ===================