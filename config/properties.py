import os

cwd = r"C:\Users\hp\Documents\WQU\Capstone\Risk_and_Volatility_Project"

# For ETL processes
etl_config_path = os.path.join(cwd, "config", "etl_config.json")
output_folder = os.path.join(cwd, "data")
market_data_output_folder = os.path.join(output_folder, "market_data")

# For Modelling processes
model_config_path = os.path.join(cwd, "config", "model_config.json")
train_set_path = os.path.join(market_data_output_folder, "train_set.csv")
test_set_path = os.path.join(market_data_output_folder, "test_set.csv")
model_prediction_path = os.path.join(output_folder, "model_predictions")
model_path = os.path.join(output_folder, "models")

# For Model Evaluation
model_evaluation_report_path = os.path.join(output_folder, "model_evaluation")
model_evaluation_report_raw_path = os.path.join(model_evaluation_report_path, "raw")
model_evaluation_report_compiled_path = os.path.join(model_evaluation_report_path, "compiled")
backtest_recent_path = os.path.join(model_evaluation_report_path, "backtest", "recent")
backtest_stress_path = os.path.join(model_evaluation_report_path, "backtest", "stress")
shap_path = os.path.join(model_evaluation_report_path, "shap")
