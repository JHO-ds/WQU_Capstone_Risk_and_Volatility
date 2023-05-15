import time
from src.models.logistic_regression import Logistic_Regression
from src.models.random_forest import RandomForest
from src.models.lightgbm_classifier import LightGBM
from src.models.xgboost_classifier import XGBoost
from src.models.ANN import ANN
from src.models.LSTM import LongShortTM
from src.models.ensemble_voting import Ensemble_Voting
from src.models.ensemble_stacking import Ensemble_Stacking

if __name__ == "__main__":

    # 1. Logistic Regression
    start = time.time()
    log_model = Logistic_Regression()
    # log_model.generate_shap_value()
    log_train_time = time.time() - start

    # 3. Random-Forest
    start = time.time()
    rf_model = RandomForest()
    rf_model.generate_shap_value()
    rf_train_time = time.time() - start

    # 4. Boosting
    start = time.time()
    lightgbm_model = LightGBM()
    lightgbm_model.generate_shap_value()
    lightgbm_train_time = time.time() - start

    start = time.time()
    xgboost_model = XGBoost()
    xgboost_model.generate_shap_value()
    xgboost_train_time = time.time() - start

    # 5. Neural Network - ANN
    start = time.time()
    ann_model = ANN()
    ann_model.generate_shap_value()
    ann_train_time = time.time() - start

    # 6. Neural Network - LSTM
    start = time.time()
    lstm_model = LongShortTM()
    lstm_model.generate_shap_value()
    lstm_train_time = time.time() - start

    # 7. Ensembly of Models
    start = time.time()
    ensemble_voting_model = Ensemble_Voting()
    ensemble_voting_model.generate_shap_value()
    voting_train_time = time.time() - start

    start = time.time()
    ensemble_stacking_model = Ensemble_Stacking()
    ensemble_stacking_model.generate_shap_value()
    stacking_train_time = time.time() - start

    print(f"""
    The time taken (s) to generate SHAP Value for the following are noted below:
    Logistic        : {log_train_time:.2f}
    Random Forest   : {rf_train_time:.2f}
    LightGBM        : {lightgbm_train_time:.2f}
    XGBoost         : {xgboost_train_time:.2f}
    ANN             : {ann_train_time:.2f}
    LSTM            : {lstm_train_time:.2f}
    Voting          : {voting_train_time:.2f}
    Stacking        : {stacking_train_time:.2f}
    """)


