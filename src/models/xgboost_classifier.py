import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import os
import copy
import warnings

from config import properties as p
from src.models.model_main import Models
from src.model_evaluation.evaluation_metrics import classification_score


warnings.filterwarnings("ignore")


class XGBoost(Models):

    def __init__(self):
        super().__init__()
        self.curr_config = self.config.get("XGBoost")
        self.model_name = self.curr_config.get("name")
        self.eval_metric = self.curr_config.get("evaluation_metric", "accuracy")

    def data_preprocessing(self):
        x_column = self.curr_config.get("explanatory_variables")
        y_column = self.curr_config.get("target_variable")

        # convert to categorical
        y_train = pd.Categorical(self.train_set[y_column])
        y_test = pd.Categorical(self.test_set[y_column])

        x_train, y_train = np.array(self.train_set[x_column]), np.array(y_train)
        x_test, y_test = np.array(self.test_set[x_column]), np.array(y_test)

        x_train, y_train = x_train.reshape(-1, len(x_column)), y_train
        x_test, y_test = x_test.reshape(-1, len(x_column)), y_test

        return x_train, y_train, x_test, y_test

    def create_random_grid(self) -> dict:
        """
        Refer to - https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
        """
        y_column = self.curr_config.get("target_variable")
        no_of_pos_class = sum(self.train_set[y_column] == 1)
        no_of_neg_class = sum(self.train_set[y_column] == 0)
        if no_of_neg_class > no_of_pos_class:
            weight_scale = max(1, round(no_of_neg_class/no_of_pos_class))
        else:
            weight_scale = min(1, round(no_of_neg_class / no_of_pos_class))

        parameter_grid = {
            "n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=10)],
            "max_depth": [int(x) for x in np.linspace(5, 30, num=10)],
            "max_leaves": [2, 5, 10],
            "grow_policy": ["depthwise", "lossguide"],
            "tree_method": ["approx", "hist", "gpu_hist"],
            "learning_rate": [0.1],
            "objective": ["binary:logistic"],
            "booster": ["gbtree", "gblinear"],
            "gamma": [float(x) for x in np.linspace(0.5, 1.5, num=6)],
            "min_child_weight": [1],
            "max_delta_step": [0],
            "subsample":  [0.4],
            "sampling_method": ["uniform", "gradient_based"],
            "colsample_bytree":  [0.4],
            "colsample_bylevel":  [0.4],
            "colsample_bynode":  [0.4],
            "reg_alpha": [float(x) for x in np.linspace(0.20, 1, num=10)] + [2, 5, 10],
            "reg_lambda": [float(x) for x in np.linspace(0.20, 1, num=10)] + [2, 5, 10],
            "scale_pos_weight": [weight_scale],
            "random_state": [self.random_state]
        }
        return parameter_grid

    def create_grid(self, best_random_param: dict) -> dict:
        grid_search = copy.deepcopy(best_random_param)

        # generate a linear space for hyperparameter tuning
        grid_search["max_depth"] = [int(x) for x in range(5, 30)]
        # grid_search["n_estimators"] = [int(x) for x in np.linspace(start=10,
        #                                                            stop=50,
        #                                                            num=4)]
        # grid_search["min_child_weight"] = [2]

        for key, val in grid_search.items():
            if not isinstance(grid_search[key], list):
                grid_search[key] = [val]
        return grid_search


    def model_training(self):
        x_train, y_train, x_test, y_test = self.data_preprocessing()

        # train the model via hyperparameter tuning

        # 1. Random Search
        random_grid = self.create_random_grid()
        self.base_model = XGBClassifier()
        self.random_search_model = RandomizedSearchCV(
            estimator=self.base_model,
            param_distributions=random_grid,
            scoring=self.eval_metric,
            n_iter=100, cv=3, verbose=2, n_jobs=-1
        )

        self.random_search_model.fit(x_train, y_train)

        random_search_param = self.random_search_model.best_params_

        # 2. Grid Search - Narrow down the search by using Random Search Best Parameters
        optimized_grid = self.create_grid(random_search_param)
        self.base_model = XGBClassifier()
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=optimized_grid,
            scoring=self.eval_metric,
            cv=3, verbose=2, n_jobs=-1
        )
        self.model.fit(x_train, y_train)

        grid_search_param = self.model.best_params_

        print(f"The random search parameter: {random_search_param}")
        print(f"The grid search parameter: {grid_search_param}")

        # predict the probabilities
        y_pred = self.model.predict_proba(x_test)

        # generate the predicted class based on the benchmark
        y_pred_df = pd.DataFrame(y_pred, columns=["Class 0", "Class 1"], index=self.test_set["Date"])
        benchmark = self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["Class 1"].apply(lambda x: int(x >= benchmark))

        # output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))
