import pandas as pd
import numpy as np
from sklearn import linear_model
import os

from config import properties as p
from src.models.model_main import Models
from src.model_evaluation.evaluation_metrics import classification_score


class Logistic_Regression(Models):

    def __init__(self):
        super().__init__()
        self.curr_config = self.config.get("Logistic_Regression")
        self.model_name = self.curr_config.get("name")

    def data_preprocessing(self):
        x_column = self.curr_config.get("explanatory_variables")
        y_column = self.curr_config.get("target_variable")

        x_train, y_train = np.array(self.train_set[x_column]), np.array(self.train_set[y_column])
        x_test, y_test = np.array(self.test_set[x_column]), np.array(self.test_set[y_column])

        x_train, y_train = x_train.reshape(-1, len(x_column)), y_train
        x_test, y_test = x_test.reshape(-1, len(x_column)), y_test

        return x_train, y_train, x_test, y_test

    def model_training(self):
        x_train, y_train, x_test, y_test = self.data_preprocessing()

        # train the model
        self.model = linear_model.LogisticRegression()
        self.model.fit(x_train, y_train)

        # predict the probabilities
        y_pred = self.model.predict_proba(x_test)

        # generate the predicted class based on the benchmark
        y_pred_df = pd.DataFrame(y_pred, columns=["Class 0", "Class 1"], index=self.test_set["Date"])
        benchmark = self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["Class 1"].apply(lambda x: int(x >= benchmark))

        # output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))
