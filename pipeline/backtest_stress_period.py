from src.models.logistic_regression import Logistic_Regression
from src.models.random_forest import RandomForest
from src.models.lightgbm_classifier import LightGBM
from src.models.xgboost_classifier import XGBoost
from src.models.ensemble_voting import Ensemble_Voting
from src.models.ensemble_stacking import Ensemble_Stacking
from src.model_evaluation.evaluation_metrics import compile_model_eval_reports

if __name__ == "__main__":
    log_model = Logistic_Regression()
    log_model.model_evaluation(backtest="stress")
    log_model.backtest_strategy(backtest="stress")

    rf_model = RandomForest()
    rf_model.model_evaluation(backtest="stress")
    rf_model.backtest_strategy(backtest="stress")

    lightgbm_model = LightGBM()
    lightgbm_model.model_evaluation(backtest="stress")
    lightgbm_model.backtest_strategy(backtest="stress")

    xgboost_model = XGBoost()
    xgboost_model.model_evaluation(backtest="stress")
    xgboost_model.backtest_strategy(backtest="stress")

    ensemble_voting_model = Ensemble_Voting()
    ensemble_voting_model.model_evaluation(backtest="stress")
    ensemble_voting_model.backtest_strategy(backtest="stress")

    ensemble_stacking_model = Ensemble_Stacking()
    ensemble_stacking_model.model_evaluation(backtest="stress")
    ensemble_stacking_model.backtest_strategy(backtest="stress")

    compile_model_eval_reports()
