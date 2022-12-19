import time
import pandas as pd
import tensorflow.keras.preprocessing as data_preprocessor
import nltk
import contractions
from lib.text_removal import TextRemoval
from nltk.tokenize import word_tokenize

def snowball_stemmer(text):
    '''
        Stem words in list of tokenized words with SnowballStemmer
    '''
    stemmer = nltk.SnowballStemmer("english")
    stems = [stemmer.stem(i) for i in text]
    return stems

def miss_value_evaluate(pandas_dataframe):
    '''
        評估DataFrame中各特徵的缺失值佔比
    '''
    evaluate_result = pandas_dataframe.isnull().sum() / len(pandas_dataframe)
    print(evaluate_result, "\n")

class TextPreprocessing():
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.tokenizer = None

    def view_datasets(self):
        '''
        分析資料內容
        '''
        # 資料外觀與數量
        print("Training data :")
        print(self.df_train[~self.df_train["location"].isnull()].head(5))
        print(self.df_train.shape, "\n")
        print("Testing data :")
        print(self.df_test[~self.df_test["location"].isnull()].head(5))
        print(self.df_test.shape, "\n")

        return True

    def import_datasets(self, train_path, test_path):
        '''
        資料匯入
        '''
        print("\nStart importing datasets . . .")
        print("========================================")
        print("========================================")
        # 建立 DataFrame
        print("Transforming datasets to DataFrame . . .\n")
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)

        # 評估缺失值
        print("========================================")
        print("Evaluating percentage of missing value . . .\n")
        print("Training data :")
        miss_value_evaluate(self.df_train)
        print("Testing data :")
        miss_value_evaluate(self.df_test)
        
        # 審視資料內容
        print("========================================")
        print("Viewing datasets . . .\n")
        self.view_datasets()

        # 提取重點資料
        print("========================================")
        print("Extracting needed data from datasets . . .")
        self.df_train = self.df_train.loc[:, ['id', 'text', 'target']]
        self.df_test = self.df_test.loc[:, ['id', 'text']]

        print("========================================")
        print("========================================")
        print("Importing success ! ! !\n")
        time.sleep(5.0)
        return True
    
    def text_cleaning(self):
        '''
        資料清理
        '''
        print("\nStart cleaning text data . . .")
        print("========================================")
        print("========================================")
        print("原始文本 : ")
        print("===>>> ", self.df_train['text'].values[33], "\n")

        # 文本小寫化
        print("文本小寫化 . . .\n")
        self.df_train["text_clean"] = self.df_train["text"].apply(lambda x: x.lower())
        self.df_test["text_clean"] = self.df_test["text"].apply(lambda x: x.lower())
        print("===>>> ", self.df_train['text_clean'].values[33])
        # 文本縮詞展開
        print("========================================")
        print("文本縮詞展開 . . .\n")
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: contractions.fix(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: contractions.fix(x))
        print("===>>> ", self.df_train['text_clean'].values[33])
        # 文本雜訊過濾 - URLs(網址)
        print("========================================")
        print("文本雜訊過濾 - URLs(網址) . . .\n")
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_URL(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_URL(x))
        print("===>>> ", self.df_train['text_clean'].values[33])
        # 文本雜訊過濾 - HTML_tags(HTML標籤)
        print("========================================")
        print("文本雜訊過濾 - HTML_tags(HTML標籤) . . .\n")
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_html(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_html(x))
        print("===>>> ", self.df_train['text_clean'].values[33])
        # 文本雜訊過濾 - Non_ASCII(非ASCII字元)
        print("========================================")
        print("文本雜訊過濾 - Non_ASCII(非ASCII字元) . . .\n")
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_non_ascii(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_non_ascii(x))
        print("===>>> ", self.df_train['text_clean'].values[33])
        # 文本雜訊過濾 - punctuations(標點符號)
        print("========================================")
        print("文本雜訊過濾 - punctuations(標點符號) . . . \n")
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.remove_punct(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.remove_punct(x))
        print("===>>> ", self.df_train['text_clean'].values[33])
        # 文本雜訊過濾 - 替換拼寫錯誤、俚語、首字母縮略詞或非正式縮寫
        print("========================================")
        print("文本雜訊過濾 - 替換拼寫錯誤、俚語、首字母縮略詞或非正式縮寫 . . .\n")
        self.df_train["text_clean"] = self.df_train["text_clean"].apply(lambda x: TextRemoval.other_clean(x))
        self.df_test["text_clean"] = self.df_test["text_clean"].apply(lambda x: TextRemoval.other_clean(x))
        print("===>>> ", self.df_train['text_clean'].values[33])

        print("========================================")
        print("========================================")
        print("Cleaning success ! ! !\n")
        time.sleep(5.0)
        return True

    def text_tokenize(self):
        '''
        資料標記
        '''
        print("\nStart tokenizing text data . . .")
        print("========================================")
        print("========================================")

        # 文本斷詞
        print("Segmenting texts . . .\n")
        nltk.download('punkt')
        self.df_train['tokenized'] = self.df_train["text_clean"].apply(word_tokenize)
        self.df_test['tokenized'] = self.df_test["text_clean"].apply(word_tokenize)
        print("===>>> ", self.df_train['tokenized'].values[33])

        # 文本數值化
        print("========================================")
        print("Digitizing texts . . .\n")
        self.tokenizer = data_preprocessor.text.Tokenizer(num_words=10000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.df_train["tokenized"])
        self.df_train["tokenized"] = self.tokenizer.texts_to_sequences(self.df_train["tokenized"])
        self.df_test["tokenized"] = self.tokenizer.texts_to_sequences(self.df_test["tokenized"])
        print("===>>> ", self.df_train['tokenized'].values[33])
        
        print("========================================")
        print("========================================")
        print("Tokenizing success ! ! !\n")
        time.sleep(5.0)
        return True
        
        """
        # Remove Stop Words
        nltk.download('stopwords')
        stop = set(stopwords.words('english'))
        self.df_train['stopwords_removed'] = self.df_train["tokenized"].apply(lambda x: [word for word in x if word not in stop])
        # Stemming
        self.df_train['snowball_stemmer'] = self.df_train['stopwords_removed'].apply(lambda x: snowball_stemmer(x))
        # POS tagging
        """

    def load_datasets(self, train_url, test_url):
        '''
        載入資料並回傳已清理且標記完的資料
        '''
        self.import_datasets(train_url, test_url)
        self.text_cleaning()
        self.text_tokenize()

        return self.df_train, self.df_test, self.tokenizer
