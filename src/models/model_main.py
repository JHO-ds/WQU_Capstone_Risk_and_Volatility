import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import roc_auc_score

from config import properties as p
from src.utils import utils as util
from src.model_evaluation.evaluation_metrics import aic_calc, bic_calc, classification_score
from src.model_evaluation.evaluation_metrics import portfolio_evaluation


class Models:

    def __init__(self):
        self.train_set = pd.read_csv(p.train_set_path)
        self.test_set = pd.read_csv(p.test_set_path)
        self.config = util.read_json(p.model_config_path)
        self.random_state = 42

    def model_evaluation(self):
        self.model = pickle.load(open(os.path.join(p.model_path, f"{self.model_name}.pkl"), "rb"))
        x_train, y_train, x_test, y_test = self.data_preprocessing()
        benchmark = self.curr_config.get("prediction_benchmark")

        # 1. evaluate model performance on train set
        train_predicted_prob = self.model.predict_proba(x_train)
        train_predicted_prob = train_predicted_prob[:, 1]
        aic_train = aic_calc(y_train, train_predicted_prob, x_train.shape[1])
        bic_train = bic_calc(y_train, train_predicted_prob, x_train.shape[1])
        roc_auc_train = roc_auc_score(y_train, train_predicted_prob)
        train_prediction = np.vectorize(util.map_class)(train_predicted_prob, benchmark)
        print(f"Evaluating {self.model_name} - Train Set :: ")
        res_train, acc_train, precision_train, recall_train, f1_train, cm_train = classification_score(train_prediction,
                                                                                                       y_train,
                                                                                                       roc_auc_train)

        train_report = pd.DataFrame({
            self.model_name: {
                "Train_Accuracy": acc_train,
                "Train_Precision": precision_train,
                "Train_Recall": recall_train,
                "Train_F1-Score": f1_train,
                "Train_AIC": aic_train,
                "Train_BIC": bic_train,
                "Train_ROC_AUC_Score": roc_auc_train
            }
        }).fillna(float(0))
        train_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_train_report.csv"))

        # evaluate model performance on test set
        test_predicted_prob = self.model.predict_proba(x_test)
        test_predicted_prob = test_predicted_prob[:, 1]
        aic_test = aic_calc(y_test, test_predicted_prob, x_test.shape[1])
        bic_test = bic_calc(y_test, test_predicted_prob, x_test.shape[1])
        roc_auc_test = roc_auc_score(y_test, test_predicted_prob)
        test_prediction = np.vectorize(util.map_class)(test_predicted_prob, benchmark)
        print(f"Evaluating {self.model_name} - Test Set :: ")
        res_test, acc_test, precision_test, recall_test, f1_test, cm_test = classification_score(test_prediction,
                                                                                                 y_test,
                                                                                                 roc_auc_test)

        test_report = pd.DataFrame({
            self.model_name: {
                "Test_Accuracy": acc_test,
                "Test_Precision": precision_test,
                "Test_Recall": recall_test,
                "Test_F1-Score": f1_test,
                "Test_AIC": aic_test,
                "Test_BIC": bic_test,
                "Test_ROC_AUC_Score": roc_auc_test
            }
        }).fillna(float(0))
        test_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_test_report.csv"))


    def backtest_strategy(self, transactional_cost: float = 0):
        ret_colname = util.read_json(p.etl_config_path).get("data_loading").get("EQ_Ticker_Name")
        return_series = self.test_set[ret_colname]
        y_pred_path = os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv")
        y_pred = pd.read_csv(y_pred_path)["final_pred"]
        predicted_returns = y_pred * return_series - y_pred * transactional_cost
        print(f"Backtesting {self.model_name} strategy :: ")
        backtest_result = portfolio_evaluation(
            portfolio_returns=pd.Series(predicted_returns),
            benchmark_returns=pd.Series(return_series),
            rf=5.0
        )
        backtest_result.columns = [self.model_name]
        backtest_result.to_csv(os.path.join(p.backtest_recent_path, f"{self.model_name}_backtest_recent.csv"))


    def save_model(self):
        with open(os.path.join(p.model_path, f'{self.model_name}.pkl'), 'wb') as file:
            pickle.dump(self.model, file)
