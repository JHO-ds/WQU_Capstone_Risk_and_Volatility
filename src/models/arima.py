import pandas as pd
import numpy as np
import pmdarima as pm
import pickle
import os

from config import properties as p
from src.utils import utils as util
from src.models.model_main import Models
from src.model_evaluation.evaluation_metrics import aic_calc, bic_calc, classification_score


class ARIMA(Models):

    def __init__(self):
        super().__init__()
        self.curr_config = self.config.get("ARIMA")
        self.model_name = self.curr_config.get("name")

    def data_preprocessing(self):
        y_column = self.curr_config.get("time-series")
        x_column = self.curr_config.get("exogenous")
        time_series_data_train = np.asarray(self.train_set[y_column]).reshape(-1, 1)
        exogenous_data_train = np.asarray(self.train_set[x_column]).reshape(-1, 1)
        exogenous_data_test = np.asarray(self.test_set[x_column]).reshape(-1, 1)
        return time_series_data_train, exogenous_data_train, exogenous_data_test

    def model_training(self):
        time_series_data, exogenous_data_train, exogenous_data_test = self.data_preprocessing()
        self.model = pm.auto_arima(
            time_series_data, exogenous=exogenous_data_train,
            start_p=1, start_q=1,
            test='adf',
            max_p=3, max_q=3, m=1,
            start_P=0, seasonal=False,
            d=None, D=0, trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        # generate forcast
        y_pred, confidence_interval = self.model.predict(n_periods=self.test_set.shape[0],
                                                         exogenous=exogenous_data_test,
                                                         return_conf_int=True)

        # transform the forecast into price signals
        y_pred_df = pd.DataFrame(y_pred, columns=["forecast"], index=self.test_set["Date"])
        benchmark = self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["forecast"].apply(lambda x: int(x >= benchmark))

        # output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))


    def model_evaluation(self):
        self.model = pickle.load(open(os.path.join(p.model_path, f"{self.model_name}.pkl"), "rb"))
        time_series_data, exogenous_data_train, exogenous_data_test = self.data_preprocessing()
        benchmark = self.curr_config.get("prediction_benchmark")

        # 1. evaluate model performance on train set
        train_predicted = self.model.arima_res_.fittedvalues
        aic_train = self.model.aic()
        bic_train = self.model.bic()
        train_prediction = np.vectorize(util.map_class)(train_predicted, benchmark)
        res_train, acc_train, precision_train, recall_train, f1_train, cm_train = classification_score(train_prediction,
                                                                                                       self.train_set["target"],
                                                                                                       0)

        train_report = pd.DataFrame({
            self.model_name: {
                "Train_Accuracy": acc_train,
                "Train_Precision": precision_train,
                "Train_Recall": recall_train,
                "Train_F1-Score": f1_train,
                "Train_AIC": aic_train,
                "Train_BIC": bic_train
            }
        }).fillna(float(0))
        train_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_train_report.csv"))

        # evaluate model performance on test set
        y_pred, confidence_interval = self.model.predict(n_periods=self.test_set.shape[0],
                                                         exogenous=exogenous_data_test,
                                                         return_conf_int=True)
        y_pred_df = pd.DataFrame(y_pred, columns=["forecast"], index=self.test_set["Date"])
        y_pred_df["final_pred"] = y_pred_df["forecast"].apply(lambda x: int(x >= benchmark))
        y_column = self.curr_config.get("time-series")
        k = self.model.params().shape[0]
        aic_test = aic_calc(self.test_set[y_column], y_pred_df["forecast"], k, "regression")
        bic_test = bic_calc(self.test_set[y_column], y_pred_df["forecast"], k, "regression")
        res_test, acc_test, precision_test, recall_test, f1_test, cm_test = classification_score(y_pred_df["final_pred"],
                                                                                                 self.test_set["target"],
                                                                                                 0)

        test_report = pd.DataFrame({
            self.model_name: {
                "Test_Accuracy": acc_test,
                "Test_Precision": precision_test,
                "Test_Recall": recall_test,
                "Test_F1-Score": f1_test,
                "Test_AIC": aic_test,
                "Test_BIC": bic_test
            }
        }).fillna(float(0))
        test_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_test_report.csv"))
