import pandas as pd
import numpy as np
import os

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

from config import properties as p
from src.models.model_main import Models
from src.models.nn_main import NN_main

import warnings
warnings.filterwarnings("ignore") # need to import warnings as it will flag up


class ANN(Models, NN_main):

    def __init__(self):
        super().__init__()
        self.curr_config = self.config.get("ANN")
        self.model_name = self.curr_config.get("name")
        self.eval_metric = self.curr_config.get("evaluation_metric", "accuracy")
        self.activationList = ["relu", "sigmoid", "tanh"]
        self.batch_sizeList = [2 ** x for x in range(3, 7)]
        self.epochesList = [x * 50 for x in range(2, 5)]
        tf.random.set_seed(self.random_state)

    def data_preprocessing(self):
        x_column = self.curr_config.get("explanatory_variables")
        y_column = self.curr_config.get("target_variable")

        x_train, y_train = np.array(self.train_set[x_column]), np.array(self.train_set[y_column])
        x_test, y_test = np.array(self.test_set[x_column]), np.array(self.test_set[y_column])

        x_train, y_train = x_train.reshape(-1, len(x_column)), y_train.reshape((-1, 1))
        x_test, y_test = x_test.reshape(-1, len(x_column)), y_test.reshape((-1, 1))

        return x_train, y_train, x_test, y_test

    def predict_probability(self, input_data: np.array) -> np.array:
        return self.model.predict(input_data).reshape(-1)

    def nn_cl_bo(self, layer1, activation, batch_size, epoches):
        x_train, y_train, x_test, y_test = self.data_preprocessing()
        layer1 = round(layer1)
        activation = self.activationList[round(activation)]
        batch_size = self.batch_sizeList[round(batch_size)]
        epoches = self.epochesList[round(epoches)]

        # Redefine a function. Tried to redefine outside but it only can take layer1 and
        # activation as a variable.
        def nn_cl_fun():
            model = Sequential()
            model.add(Dense(layer1,
                            input_shape=(x_train.shape[1], ),
                            activation=activation))
            model.add(Dense(1,
                            activation='sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[self.metrics_keras(self.eval_metric)])
            return model

        es = EarlyStopping(monitor=self.early_stopping_metrics(self.eval_metric),
                           mode='max',
                           verbose=0,
                           patience=20)

        nn = KerasClassifier(build_fn=nn_cl_fun,
                             epochs=epoches,
                             batch_size=batch_size,
                             verbose=0)

        score_acc = make_scorer(self.metrics_sklearn(self.eval_metric))
        score = cross_val_score(nn,
                                x_train,
                                y_train,
                                verbose=0,
                                scoring=score_acc,
                                fit_params={'callbacks': [es]}).mean()

        return score

    def model_training(self):
        # Not sure if the training set can be fed into the function directly.
        x_train, y_train, x_test, y_test = self.data_preprocessing()

        params_nn = {
            'layer1': (8, 50),
            'activation': (0, 2),
            'batch_size': (0, 3),
            'epoches': (0, 2)}

        # Create the objective function
        nn_bo = BayesianOptimization(self.nn_cl_bo,
                                     params_nn,
                                     random_state=self.random_state)

        # Maximize the objective function
        nn_bo.maximize(init_points=1, n_iter=1) #init_points=90, n_iter=10 -> if it works, please return to the original values

        # Get optimal parameters
        params_nn_ = nn_bo.max['params']
        layer1 = round(params_nn_['layer1'])
        activation = self.activationList[round(params_nn_['activation'])]
        batch_size = self.batch_sizeList[round(params_nn_['batch_size'])]
        epoches = self.epochesList[round(params_nn_['epoches'])]

        # Fit the model with the optimal parameters
        self.model = Sequential()
        self.model.add(Dense(
                units=layer1,
                activation=activation,
                input_shape=(x_train.shape[1], )))
        self.model.add(Dense(
                units=1,
                activation="sigmoid"))

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[self.metrics_keras(self.eval_metric)])

        early_stopping = EarlyStopping(monitor=self.early_stopping_metrics(self.eval_metric),
                                       mode='max',
                                       verbose=0,
                                       patience=20)

        self.model.fit(x_train,
                      y_train,
                      epochs=epoches,
                      batch_size=batch_size,
                      verbose=0,
                      callbacks=[early_stopping])

        y_pred = self.model.predict(x_test).reshape(-1)

        # generate the predicted class based on the benchmark
        y_pred_df = pd.DataFrame({
            "Class 0": 1-y_pred,
            "Class 1": y_pred
        }, index=self.test_set["Date"])
        benchmark = self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["Class 1"].apply(lambda x: int(x >= benchmark))

        # output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))
