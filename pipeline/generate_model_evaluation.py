from src.models.logistic_regression import Logistic_Regression
from src.models.random_forest import RandomForest
from src.models.lightgbm_classifier import LightGBM
from src.models.xgboost_classifier import XGBoost
from src.models.ANN import ANN
from src.models.LSTM import LongShortTM
from src.models.ensemble_voting import Ensemble_Voting
from src.models.ensemble_stacking import Ensemble_Stacking
from src.model_evaluation.evaluation_metrics import compile_model_eval_reports

if __name__ == "__main__":
    log_model = Logistic_Regression()
    log_model.model_evaluation()
    log_model.backtest_strategy()

    # arima_model = ARIMA()
    # arima_model.model_evaluation()
    # arima_model.backtest_strategy()

    rf_model = RandomForest()
    rf_model.model_evaluation()
    rf_model.backtest_strategy()

    lightgbm_model = LightGBM()
    lightgbm_model.model_evaluation()
    lightgbm_model.backtest_strategy()

    xgboost_model = XGBoost()
    xgboost_model.model_evaluation()
    xgboost_model.backtest_strategy()

    ann_model = ANN()
    ann_model.model_evaluation()
    ann_model.backtest_strategy()

    lstm_model = LongShortTM()
    lstm_model.model_evaluation()
    lstm_model.backtest_strategy()

    ensemble_voting_model = Ensemble_Voting()
    ensemble_voting_model.model_evaluation()
    ensemble_voting_model.backtest_strategy()

    ensemble_stacking_model = Ensemble_Stacking()
    ensemble_stacking_model.model_evaluation()
    ensemble_stacking_model.backtest_strategy()

    compile_model_eval_reports()
