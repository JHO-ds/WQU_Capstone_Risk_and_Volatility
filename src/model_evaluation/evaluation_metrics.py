import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_auc_score

from config import properties as p


def aic_calc(y_true: np, y_pred: np, k: int, objective: str = "classification") -> float:
    """
    This function is from the RegscorePy library - https://pypi.org/project/RegscorePy/
    We re-created the function to bypass the type error - element is not int or float:
    The function should be able to take in int64, float64, etc.
    Additionally, the function seems to be calculating AIC for linear regression.
    This function improves on the existing feature by calculating the log-likelihood for
    binary classification.
    """

    if len(y_true) != len(y_pred):
        raise Exception("y_true and y_pred should have the same length.")
    if k <= 0 or not isinstance(k, int):
        raise Exception("k refers to the number of explanatory variables. k should be a positive integer")
    if objective not in ("regression", "classification"):
        raise Exception("Objective should be regression or classification.")

    if objective == "regression":
        resid = np.subtract(y_pred, y_true)
        rss = np.sum(np.power(resid, 2))
        n = len(y_true)
        aic_score = n*np.log(rss/n) + 2*k
    elif objective == "classification":
        n = len(y_true)
        LLH = np.sum((y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)))/n
        aic_score = -2*n*LLH + 2*k
    return aic_score


def bic_calc(y_true: np, y_pred: np, k: int, objective: str = "classification") -> float:
    """
    This function is from the RegscorePy library - https://pypi.org/project/RegscorePy/
    We re-created the function to bypass the type error - element is not int or float:
    The function should be able to take in int64, float64, etc.
    Additionally, the function seems to be calculating AIC for linear regression.
    This function improves on the existing feature by calculating the log-likelihood for
    binary classification.
    """

    if len(y_true) != len(y_pred):
        raise Exception("y_true and y_pred should have the same length.")
    if k <= 0 or not isinstance(k, int):
        raise Exception("k refers to the number of explanatory variables. k should be a positive integer")
    if objective not in ("regression", "classification"):
        raise Exception("Objective should be regression or classification.")

    if objective == "regression":
        resid = np.subtract(y_pred, y_true)
        rss = np.sum(np.power(resid, 2))
        n = len(y_true)
        bic_score = n*np.log(rss/n) + k*np.log(n)
    elif objective == "classification":
        n = len(y_true)
        LLH = np.sum((y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)))/n
        bic_score = -2*n*LLH + k*np.log(n)
    return bic_score


def classification_score(y_pred: np.array, y_true: np.array, roc_auc: float, verbose: bool = True) -> tuple:
    class_1 = sum(y_true == 1)
    class_0 = sum(y_true == 0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = pd.DataFrame(
        [[tn, fn], [fp, tp]],
        columns=[("Actual", "Class 0"), ("Actual", "Class 1")],
        index=[("Predicted", "Class 0"), ("Predicted", "Class 1")]
    )
    accuracy = (tn + tp) / (tn + tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    result = """
    Model Evaluation - Classification
    =================================

    No of Class 0: {}
    No of Class 1: {}

    Confusion Matrix: 
    {}

    Classification Metrics:
    Accuracy        : {:.3f}
    Precision       : {:.3f}
    Recall          : {:.3f}
    F1-Score        : {:.3f}
    ROC_AUC Score   : {:.3f}

    """.format(
        class_0, class_1,
        cm.to_string().replace("\n", "\n\t\t\t"),
        accuracy,
        precision,
        recall,
        F1,
        roc_auc
    )
    if verbose:
        print(result)
    return result, accuracy, precision, recall, F1, cm


def compile_model_eval_reports():
    # compile train set evaluation report
    try:
        all_report_names = [elem for elem in os.listdir(p.model_evaluation_report_path) if ".csv" in elem]
        all_train_reports = [elem for elem in all_report_names if "train" in elem]
        all_train_df = [pd.read_csv(os.path.join(p.model_evaluation_report_path, elem)) for elem in all_train_reports]
        all_train_df = [df.set_index("Unnamed: 0") for df in all_train_df]
        train_reports = pd.concat(all_train_df, axis=1)
        train_reports.to_csv(os.path.join(p.model_evaluation_report_path, "compiled", "train_compiled_report.csv"))
    except Exception as e:
        print("Error occurred while compiling train set reports")

    try:
        # compile test set evaluation report
        all_test_reports = [elem for elem in all_report_names if "test" in elem]
        all_test_df = [pd.read_csv(os.path.join(p.model_evaluation_report_path, elem)) for elem in all_test_reports]
        all_test_df = [df.set_index("Unnamed: 0") for df in all_test_df]
        test_reports = pd.concat(all_test_df, axis=1)
        test_reports.to_csv(os.path.join(p.model_evaluation_report_path, "compiled", "test_compiled_report.csv"))
    except Exception as e:
        print("Error occurred while compiling test set reports")

    try:
        # compile stress evaluation report
        all_stress_reports = [elem for elem in all_report_names if "stress" in elem]
        all_stress_df = [pd.read_csv(os.path.join(p.model_evaluation_report_path, elem)) for elem in all_stress_reports]
        all_stress_df = [df.set_index("Unnamed: 0") for df in all_stress_df]
        stress_reports = pd.concat(all_stress_df, axis=1)
        stress_reports.to_csv(os.path.join(p.model_evaluation_report_path, "compiled", "stress_compiled_report.csv"))
    except Exception as e:
        print("Error occurred while compiling stress reports")

    try:
        # compile recent period recent backtest report
        all_backtest_names = [elem for elem in os.listdir(p.backtest_recent_path) if ".csv" in elem]
        all_backtest_df = [pd.read_csv(os.path.join(p.backtest_recent_path, elem)) for elem in all_backtest_names]
        all_backtest_df = [df.rename(columns={"Unnamed: 0": "Eval_metric"}) for df in all_backtest_df]
        all_backtest_df = [df.set_index("Eval_metric") for df in all_backtest_df]
        backtest_reports = pd.concat(all_backtest_df, axis=1)
        backtest_reports.to_csv(os.path.join(p.model_evaluation_report_path, "compiled", "backtest_recent_compiled.csv"))
    except Exception as e:
        print("Error occurred while compiling recent backtest reports")

    try:
        # compile recent period stress backtest report
        all_backtest_names = [elem for elem in os.listdir(p.backtest_stress_path) if ".csv" in elem]
        all_backtest_df = [pd.read_csv(os.path.join(p.backtest_stress_path, elem)) for elem in all_backtest_names]
        all_backtest_df = [df.rename(columns={"Unnamed: 0": "Eval_metric"}) for df in all_backtest_df]
        all_backtest_df = [df.set_index("Eval_metric") for df in all_backtest_df]
        backtest_reports = pd.concat(all_backtest_df, axis=1)
        backtest_reports.to_csv(os.path.join(p.model_evaluation_report_path, "compiled", "backtest_stress_compiled.csv"))
    except Exception as e:
        print("Error occurred while compiling stres backtest reports")


def Sharpe_Ratio(portfolio_returns: pd.Series, rf: float, n: int) -> float:
    annualized_excess_return = np.mean(portfolio_returns)*n - rf
    annualized_std = np.std(portfolio_returns) * np.sqrt(n)
    return annualized_excess_return/annualized_std


def Information_Ratio(portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series, n: int) -> float:
    excess_returns = np.mean(portfolio_returns - benchmark_returns)*n
    std_excess = np.std(portfolio_returns - benchmark_returns)*np.sqrt(n)
    return excess_returns/std_excess


def Sortino_Ratio(portfolio_returns: pd.Series, rf: float, n: int) -> float:
    negative_returns = portfolio_returns.loc[portfolio_returns<0]
    return (np.mean(portfolio_returns)*n - rf)/(np.std(negative_returns)*np.sqrt(n))


def Maximum_Drawdown(portfolio_cummulative_returns: pd.Series, window: int = 14) -> float:
    rolling_min = portfolio_cummulative_returns.rolling(window, min_periods=1).min()
    rolling_max = portfolio_cummulative_returns.rolling(window, min_periods=1).max()
    drawdown = rolling_max - rolling_min
    return max(drawdown)


def Volatility(portfolio_returns: pd.Series, n: int) -> float:
    return np.std(portfolio_returns)*np.sqrt(n)


def portfolio_evaluation(portfolio_returns: pd.Series, benchmark_returns: pd.Series, rf: float) -> pd.DataFrame:
    n = 252 # number of business days in a year

    portfolio_cumulative_returns = np.cumsum(portfolio_returns)
    benchmark_cumulative_returns = np.cumsum(benchmark_returns)

    portfolio_Sharpe_Ratio = Sharpe_Ratio(portfolio_returns, rf, n)
    portfolio_Information_Ratio = Information_Ratio(portfolio_returns, benchmark_returns, n)
    portfolio_Sortino_Ratio = Sortino_Ratio(portfolio_returns, rf, n)

    portfolio_max_drawdown = Maximum_Drawdown(portfolio_cumulative_returns)
    portfolio_vol = Volatility(portfolio_returns, n)

    print(f"""
    Portfolio Evaluation
    ====================
    Accumulated Returns : {portfolio_cumulative_returns.values.tolist()[-1]:.3f} %
    Sharpe Ratio        : {portfolio_Sharpe_Ratio:.3f}
    Information Ratio   : {portfolio_Information_Ratio:.3f}
    Sortino Ratio       : {portfolio_Sortino_Ratio:.3f}
    Maximum Drawdown    : {portfolio_max_drawdown:.3f} %
    Volatility          : {portfolio_vol:.3f}
    """)

    backtest_results = pd.DataFrame({
        "Accumulated_Returns_%": [portfolio_cumulative_returns.values.tolist()[-1]],
        "Sharpe_Ratio": [portfolio_Sharpe_Ratio],
        "Information_Ratio": [portfolio_Information_Ratio],
        "Sortino_Ratio": [portfolio_Sortino_Ratio],
        "Maximum_Drawdown": [portfolio_max_drawdown],
        "Volatility": [portfolio_vol]
    }).T

    return backtest_results