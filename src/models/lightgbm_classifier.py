import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import os
import copy

from config import properties as p
from src.models.model_main import Models
from src.model_evaluation.evaluation_metrics import classification_score


class LightGBM(Models):

    def __init__(self):
        super().__init__()
        self.curr_config = self.config.get("LightGBM")
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
        Refer to - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier
        """

        parameter_grid = {
            "boosting_type": ["gbdt"],
            "tree_learner": ["serial", "feature", "data", "voting"],
            "extra_trees": [True, False],
            "num_leaves": [int(x) for x in np.linspace(6, 50, num=5)],
            "max_depth": [int(x) for x in np.linspace(5, 30, num=10)],
            "learning_rate": [0.1],
            "n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=10)],
            "objective": ["binary"],
            "class_weight": [None, "balanced"],
            "min_split_gain": [0],
            "min_child_weight": [1e-3],
            "min_child_samples": [int(x) for x in np.linspace(start=20, stop=50, num=10)],
            "feature_pre_filter": [False, True],
            "subsample": [0.4],
            "subsample_freq": [int(x) for x in np.linspace(0, 10, num=11)],
            "colsample_bytree": [0.4],
            "colsample_bynode": [0.4],
            "reg_alpha": [float(x) for x in np.linspace(0.20, 1, num=10)] + [2, 5, 10],
            "reg_lambda": [float(x) for x in np.linspace(0.20, 1, num=10)] + [2, 5, 10],
            "random_state": [self.random_state]
        }
        return parameter_grid

    def create_grid(self, best_random_param: dict) -> dict:
        grid_search = copy.deepcopy(best_random_param)

        # generate a linear space for hyperparameter tuning
        # grid_search["num_leaves"] = [int(x) for x in np.linspace(start=max(1, grid_search["num_leaves"]-5),
        #                                                          stop=grid_search["num_leaves"] + 5,
        #                                                          num=5)]
        grid_search["max_depth"] = [int(x) for x in range(5, 30)]

        for key, val in grid_search.items():
            if not isinstance(grid_search[key], list):
                grid_search[key] = [val]
        return grid_search


    def model_training(self):
        x_train, y_train, x_test, y_test = self.data_preprocessing()

        # train the model via hyperparameter tuning

        # 1. Random Search
        random_grid = self.create_random_grid()
        self.base_model = LGBMClassifier()
        self.random_search_model = RandomizedSearchCV(
            estimator=self.base_model,
            param_distributions=random_grid,
            scoring=self.eval_metric,
            n_iter=100, cv=3, verbose=2, n_jobs=-1
        )

        self.random_search_model.fit(x_train, y_train)

        random_search_param = self.random_search_model.best_params_
        print(f"The random search parameter: {random_search_param}")

        # 2. Grid Search - Narrow down the search by using Random Search Best Parameters
        optimized_grid = self.create_grid(random_search_param)
        self.base_model = LGBMClassifier()
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=optimized_grid,
            scoring=self.eval_metric,
            cv=3, verbose=2, n_jobs=-1
        )
        self.model.fit(x_train, y_train)

        grid_search_param = self.model.best_params_
        print(f"The grid search parameter: {grid_search_param}")

        # predict the probabilities
        y_pred = self.model.predict_proba(x_test)

        # generate the predicted class based on the benchmark
        y_pred_df = pd.DataFrame(y_pred, columns=["Class 0", "Class 1"], index=self.test_set["Date"])
        benchmark = self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["Class 1"].apply(lambda x: int(x >= benchmark))

        # output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))
