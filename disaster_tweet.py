import pandas as pd
from text_learning import TextLearning

# 匯入已訓練完成的模型
TextLearned = TextLearning()
model = TextLearned.model_training()
# 將實際測試資料輸入模型進行預測
test_data = TextLearned.get_test_data()
test_pred = model.predict(test_data)
# 最後將預測結果匯入Excel檔案
submission = pd.DataFrame(test_pred)
submission.to_csv("test_prediction.csv")