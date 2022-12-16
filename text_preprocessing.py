import pandas as pd
import tensorflow.keras.preprocessing as data_preprocessor
import nltk
import contractions
from lib.text_removal import TextRemoval
from nltk.tokenize import word_tokenize

def miss_value_evaluate(pandas_dataframe):
    '''
    評估DataFrame中各特徵的缺失值佔比
    '''
    evaluate_result = pandas_dataframe.isnull().sum() / len(pandas_dataframe)
    print(evaluate_result)

class TextPreprocessing():
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.tokenizer = None

    def view_datasets_head(self):
        print(self.df_train.head(5))
        print(self.df_test.head(5))

        return True

    def import_datasets(self, train_path, test_path):
        '''
        資料匯入
        '''
        # 建立 DataFrame
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)
        # 評估缺失值
        miss_value_evaluate(self.df_train)
        miss_value_evaluate(self.df_test)
        # 檢視部分資料內容
        self.view_datasets_head()
        # 提取重點資料
        self.df_train = self.df_train.loc[:, ['id', 'text', 'target']]
        self.df_test = self.df_test.loc[:, ['id', 'text']]

        return True
    
    def text_cleaning(self):
        '''
        資料清理
        '''
        # 文本小寫化
        self.df_train["text_clean"] = self.df_train["text"].apply(lambda x: x.lower())
        self.df_test["text_clean"] = self.df_test["text"].apply(lambda x: x.lower())
        # 文本縮詞展開
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: contractions.fix(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: contractions.fix(x))
        # 文本雜訊過濾 - URLs(網址)
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_URL(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_URL(x))
        # 文本雜訊過濾 - HTML_tags(HTML標籤)
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_html(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_html(x))
        # 文本雜訊過濾 - Non_ASCII(非ASCII字元)
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_non_ascii(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_non_ascii(x))
        # 文本雜訊過濾 - punctuations(標點符號)
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_punct(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_punct(x))
        # 文本雜訊過濾 - 替換拼寫錯誤、俚語、首字母縮略詞或非正式縮寫
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.other_clean(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.other_clean(x))
        # 文本雜訊過濾 - 拼寫更正
        #df_train["text_clean"] = df_train["text_clean"].apply(lambda x: TextBlob(x).correct())

        return True

    def text_tokenize(self):
        '''
        資料標記
        '''
        # 文本斷詞
        nltk.download('punkt')
        self.df_train['tokenized'] = self.df_train["text_clean"].apply(word_tokenize)
        self.df_test['tokenized'] = self.df_test["text_clean"].apply(word_tokenize)
        """
        # Remove Stop Words
        nltk.download('stopwords')
        stop = set(stopwords.words('english'))
        self.df_train['stopwords_removed'] = self.df_train["tokenized"].apply(lambda x: [word for word in x if word not in stop])
        # Stemming
        """
        # 文本數值化
        self.tokenizer = data_preprocessor.text.Tokenizer(num_words=10000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.df_train["tokenized"])
        self.df_train["tokenized"] = self.tokenizer.texts_to_sequences(self.df_train["tokenized"])
        self.df_test["tokenized"] = self.tokenizer.texts_to_sequences(self.df_test["tokenized"])

        return True

    def load_datasets(self, train_url, test_url):
        '''
        載入資料並回傳已清理且標記完的資料
        '''
        self.import_datasets(train_url, test_url)
        self.text_cleaning()
        self.text_tokenize()

        return self.df_train, self.df_test, self.tokenizer