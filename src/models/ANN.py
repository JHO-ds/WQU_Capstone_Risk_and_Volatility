import pandas as pd
import numpy as np
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

from config import properties as p
from src.models.model_main import Models
from src.models.nn_main import NN_main


class ANN(Models, NN_main):

    def __init__(self):
        super().__init__()
        self.curr_config = self.config.get("ANN")
        self.model_name = self.curr_config.get("name")
        self.eval_metric = self.curr_config.get("evaluation_metric", "accuracy")
        tf.random.set_seed(self.random_state)

    def predict_probability(self, input_data: np.array) -> np.array:
        return self.model.predict(input_data).reshape(-1)

    def data_preprocessing(self):
        x_column = self.curr_config.get("explanatory_variables")
        y_column = self.curr_config.get("target_variable")

        x_train, y_train = np.array(self.train_set[x_column]), np.array(self.train_set[y_column])
        x_test, y_test = np.array(self.test_set[x_column]), np.array(self.test_set[y_column])

        x_train, y_train = x_train.reshape(-1, len(x_column)), y_train.reshape((-1, 1))
        x_test, y_test = x_test.reshape(-1, len(x_column)), y_test.reshape((-1, 1))

        return x_train, y_train, x_test, y_test

    def optimal_hidden_layer_nodes(self, x_train, y_train):
        # Refer to Neural Network Design: https://hagan.okstate.edu/NNDesign.pdf#page=469

        layer_1 = 50
        layer_2 = 25
        n_diff = 1

        layer_dict = {}

        for layer1 in range(layer_1 - n_diff, layer_1 + n_diff):
            for layer2 in range(layer_2 - n_diff, layer_2 + n_diff):
                model = Sequential()

                # input layer
                model.add(Dense(
                    units=layer1,
                    activation='relu',
                    input_shape=(x_train.shape[1], )))

                model.add(Dense(
                    units=layer2,
                    activation='relu',
                    input_shape=(x_train.shape[1], )))
                model.add(Dense(
                    units=1,
                    activation="sigmoid"))

                model.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=[self.metrics(self.eval_metric)])

                early_stopping = EarlyStopping(monitor='loss',
                                               patience=5,
                                               mode='min')

                model.fit(x_train,
                          y_train,
                          epochs=200,
                          batch_size=32,
                          verbose=0,
                          callbacks=[early_stopping])

                y_pred = model.predict(x_train)
                print(f"{layer1},{layer2} - {y_pred.shape}")
                y_pred = (y_pred >= self.curr_config.get("prediction_benchmark")).astype(int)

                tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
                accuracy = (tn + tp) / x_train.shape[0] * 100

                layer_nodes = str(layer1)+","+str(layer2)
                layer_dict[layer_nodes] = accuracy

        return_layer_1 = int(max(layer_dict)[:2])
        return_layer_2 = int(max(layer_dict)[-2:])

        return return_layer_1, return_layer_2

    def model_training(self):

        x_train, y_train, x_test, y_test = self.data_preprocessing()

        #layer_1, layer_2 = self.optimal_hidden_layer_nodes(x_train, y_train)

        self.model = Sequential()
        self.model.add(Dense(
                units=8,
                activation='relu',
                input_shape=(x_train.shape[1], )))
        self.model.add(Dense(
                units=8,
                activation='relu'))
        self.model.add(Dense(
                units=1,
                activation="sigmoid"))

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[self.metrics(self.eval_metric)])

        early_stopping = EarlyStopping(monitor='loss',
                                       patience=5,
                                       mode='min')

        self.model.fit(x_train,
                      y_train,
                      epochs=200,
                      batch_size=32,
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
