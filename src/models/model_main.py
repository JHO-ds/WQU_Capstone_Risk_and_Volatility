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
        self.random_state = self.config.get("random_state", 42)

    def model_evaluation(self, backtest: str = "recent"):
        model_path = os.path.join(p.model_path, f"{self.model_name}.pkl")
        self.model = util.load_model(model_path)
        x_train, y_train, x_test, y_test = self.data_preprocessing()
        benchmark = self.curr_config.get("prediction_benchmark")

        if backtest == "recent":
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

        elif backtest == "stress":
            stress_data_path = os.path.join(p.market_data_output_folder, "stress_period.csv")
            stress_data = pd.read_csv(stress_data_path)

            # extract the explanatory variables
            x_columns = self.curr_config.get("explanatory_variables")
            x_stress_data = stress_data[x_columns]
            y_column = self.curr_config.get("target_variable")
            y_stress_data = stress_data[y_column]

            stress_predicted_prob = self.model.predict_proba(x_stress_data)
            stress_predicted_prob = stress_predicted_prob[:, 1]

            # extract the return series
            aic_stress = aic_calc(y_stress_data, stress_predicted_prob, len(x_columns))
            bic_stress = bic_calc(y_stress_data, stress_predicted_prob, len(x_columns))
            roc_auc_stress = roc_auc_score(y_stress_data, stress_predicted_prob)
            stress_prediction = np.vectorize(util.map_class)(stress_predicted_prob, benchmark)
            print(f"Evaluating {self.model_name} - Stress Period :: ")
            res_stress, acc_stress, precision_stress, recall_stress, f1_stress, cm_stress = classification_score(
                stress_prediction,
                y_stress_data,
                roc_auc_stress)

            stress_report = pd.DataFrame({
                self.model_name: {
                    "Train_Accuracy": acc_stress,
                    "Train_Precision": precision_stress,
                    "Train_Recall": recall_stress,
                    "Train_F1-Score": f1_stress,
                    "Train_AIC": aic_stress,
                    "Train_BIC": bic_stress,
                    "Train_ROC_AUC_Score": roc_auc_stress
                }
            }).fillna(float(0))
            stress_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_stress_report.csv"))

        else:
            # logging statement for input error
            pass

    def backtest_strategy(self, transactional_cost: float = 0, backtest : str = "recent"):
        if backtest == "recent":
            ret_colname = util.read_json(p.etl_config_path).get("data_loading").get("EQ_Ticker_Name")
            return_series = self.test_set[ret_colname]
            y_pred_path = os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv")
            y_pred = pd.read_csv(y_pred_path)["final_pred"]
            predicted_returns = y_pred * return_series - y_pred * transactional_cost
            print(f"[Recent Period] Backtesting {self.model_name} strategy :: ")
            backtest_result = portfolio_evaluation(
                portfolio_returns=pd.Series(predicted_returns),
                benchmark_returns=pd.Series(return_series),
                rf=5.0
            )
            backtest_result.columns = [self.model_name]
            backtest_result.to_csv(os.path.join(p.backtest_recent_path, f"{self.model_name}_backtest_recent.csv"))
        elif backtest == "stress":
            # load the stress data
            stress_data_path = os.path.join(p.market_data_output_folder, "stress_period.csv")
            stress_data = pd.read_csv(stress_data_path)

            # extract the explanatory variables
            x_columns = self.curr_config.get("explanatory_variables")
            x_stress_data = stress_data[x_columns]

            # load the model and generate predictions
            self.model = util.load_model(os.path.join(p.model_path, f"{self.model_name}.pkl"))
            y_pred_prob = self.model.predict_proba(x_stress_data)[:, 1]
            benchmark = self.curr_config.get("prediction_benchmark")
            y_pred_class = np.vectorize(util.map_class)(y_pred_prob, benchmark)

            # evaluate
            ret_colname = util.read_json(p.etl_config_path).get("data_loading").get("EQ_Ticker_Name_Stress")
            return_series = stress_data[ret_colname]
            predicted_returns = y_pred_class * return_series - y_pred_class * transactional_cost
            print(f"[Stress Period] Backtesting {self.model_name} strategy :: ")
            backtest_result = portfolio_evaluation(
                portfolio_returns=pd.Series(predicted_returns),
                benchmark_returns=pd.Series(return_series),
                rf=5.0
            )
            backtest_result.columns = [self.model_name]
            backtest_result.to_csv(os.path.join(p.backtest_stress_path, f"{self.model_name}_backtest_stress.csv"))
        else:
            # include log for error message
            pass

    def save_model(self):
        with open(os.path.join(p.model_path, f'{self.model_name}.pkl'), 'wb') as file:
            pickle.dump(self.model, file)
