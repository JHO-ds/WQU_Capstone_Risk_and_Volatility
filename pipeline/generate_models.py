import time
from src.utils.utils import dir_management
from src.data.market_data_loader import Market_Data_Loader
from src.models.logistic_regression import Logistic_Regression
from src.models.random_forest import RandomForest
from src.models.lightgbm_classifier import LightGBM
from src.models.xgboost_classifier import XGBoost
from src.models.ensemble_voting import Ensemble_Voting
from src.models.ensemble_stacking import Ensemble_Stacking

if __name__ == "__main__":
    # ensure that the output folders are created
    dir_management()

    # start data loading
    start = time.time()
    loader = Market_Data_Loader()
    loader.etl_process()
    etl_load_time = time.time() - start

    # 1. Logistic Regression
    start = time.time()
    log_model = Logistic_Regression()
    log_model.model_training()
    log_model.save_model()
    log_train_time = time.time() - start

    # 2. ARIMA time-series
    # ARIMA is excluded to focus on classification algorithms
    # start = time.time()
    # arima_model = ARIMA()
    # arima_model.model_training()
    # arima_model.save_model()
    # arima_train_time = time.time() - start

    # 3. Random-Forest
    start = time.time()
    rf_model = RandomForest()
    rf_model.model_training()
    rf_model.save_model()
    rf_train_time = time.time() - start

    # 4. Boosting
    start = time.time()
    lightgbm_model = LightGBM()
    lightgbm_model.model_training()
    lightgbm_model.save_model()
    lightgbm_train_time = time.time() - start

    start = time.time()
    xgboost_model = XGBoost()
    xgboost_model.model_training()
    xgboost_model.save_model()
    xgboost_train_time = time.time() - start

    # 5. Neural Network - ANN

    # 6. Neural Network - LSTM

    # 7. Ensembly of Models
    start = time.time()
    ensemble_voting_model = Ensemble_Voting()
    ensemble_voting_model.model_training()
    ensemble_voting_model.save_model()
    voting_train_time = time.time() - start

    start = time.time()
    ensemble_stacking_model = Ensemble_Stacking()
    ensemble_stacking_model.model_training()
    ensemble_stacking_model.save_model()
    stacking_train_time = time.time() - start

    print(f"""
    The time taken (s) for the following process are noted below:
    ETL             : {etl_load_time:.2f}
    Logistic        : {log_train_time:.2f}
    Random Forest   : {rf_train_time:.2f}
    LightGBM        : {lightgbm_train_time:.2f}
    XGBoost         : {xgboost_train_time:.2f}
    Voting          : {voting_train_time:.2f}
    Stacking        : {stacking_train_time:.2f}
    """)


