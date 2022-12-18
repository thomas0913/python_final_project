import sys
import keras as ks
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing as data_preprocessor
from sklearn.model_selection import train_test_split
from text_preprocessing import TextPreprocessing
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

train_url = "./excel/train.csv"
test_url = "./excel/test.csv"
sample_submission_url = "./excel/sample_submission.csv"

class TextLearning():
    def __init__(self):
        # 資料前處理
        TextPreprocessed = TextPreprocessing()
        self.df_train,\
        self.df_test,\
        self.tokenizer\
            = TextPreprocessed.load_datasets(train_url, test_url)
        
        # 定義模型輸入資料
        self.train_data = None
        self.train_target = None
        self.valid_data = None
        self.valid_target = None

        # others
        self.MAX_SQUENCE_LENGTH = None

    def get_test_data(self):
        test_data = data_preprocessor.sequence.pad_sequences(self.df_test["tokenized"], padding='post', maxlen=self.MAX_SQUENCE_LENGTH)
        return test_data
    
    def datasets_preparing(self):
        # Zero Padding
        self.MAX_SQUENCE_LENGTH = max([len(seq) for seq in self.df_train["tokenized"]])
        data = data_preprocessor.sequence.pad_sequences(
                    self.df_train["tokenized"],
                    padding='post',
                    maxlen=self.MAX_SQUENCE_LENGTH)
        target = self.df_train['target'].values[:]

        # 資料分割
        self.train_data,\
        self.valid_data,\
        self.train_target,\
        self.valid_target\
            = train_test_split(data, target, test_size=0.10, random_state=9527)

        # 確認資料數量
        print(self.train_data.shape)
        print(self.valid_data.shape)
        print(self.train_target.shape)
        print(self.valid_target.shape)

        return True

    def build_model(self, model_select='LSTM'):
        try:
            if (model_select == 'LSTM'):
                model = ks.models.Sequential()
                model.add(layers.Embedding(self.tokenizer.num_words, 32, input_length=self.MAX_SQUENCE_LENGTH))
                model.add(layers.Bidirectional(layers.LSTM(16)))
                model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='normal'))
                model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=True), metrics=[BinaryAccuracy(threshold=0.5, name='accuracy')])
                model.summary()
                return model
            elif (model_select == 'LogisticRegression'):
                model = ks.models.Sequential()
                model.add(layers.Embedding(self.tokenizer.num_words, 32, input_length=self.MAX_SQUENCE_LENGTH))
                model.add(layers.GlobalAveragePooling1D())
                model.add(layers.Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=True), metrics=[BinaryAccuracy(threshold=0.5, name='accuracy')])
                model.summary()
                return model
            else :
                raise ValueError("ERROR: model not found.")
        except ValueError as msg:
            print(msg)
            sys.exit()

    def model_analyzing(self, model=None, fit_history=None):
        # 顯示模型評估曲線圖
        history_dict = fit_history.history
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # 生成模型視圖
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False, rankdir='LR')

    def model_training(self):
        # 準備訓練資料與驗證資料
        self.datasets_preparing()

        # 訓練模型
        model = self.build_model(model_select='LSTM')
        history = model.fit(self.train_data, self.train_target,
                            epochs=40, batch_size=512,
                            validation_data=(self.valid_data, self.valid_target),
                            verbose='auto',
                            shuffle=True)
        # 評估模型
        result_train = model.evaluate(self.train_data, self.train_target)
        result_valid = model.evaluate(self.valid_data, self.valid_target)
        print(result_train, result_valid)

        # 分析模型
        self.model_analyzing(model, history)

        return model