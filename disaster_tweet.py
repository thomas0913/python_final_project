import pandas as pd
from text_learning import TextLearning

# 匯入已訓練完成的模型
TextLearned = TextLearning()
model = TextLearned.model_training()

print("\nStart predicting test data . . .")
print("========================================")
print("========================================")
# 將實際測試資料輸入模型進行預測
print("Predicting test data form trained-model . . .\n")
test_data = TextLearned.get_test_data()
test_pred = model.predict(test_data)

# 最後將預測結果匯入Excel檔案
print("========================================")
print("Storing predicted data as an csv file . . .\n")
submission = pd.DataFrame(test_pred)
submission.to_csv("./excel/test_prediction.csv")
print("========================================")
print("========================================")
print("Predicting success ! ! !\n")
print("Thanks for your play ! ! !\n")