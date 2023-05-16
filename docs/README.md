# Capstone Project - Risk & Volatility
Authored by: Jeff Ho & Teo Chee Seong

**The aim of this project**:
1. To evaluate the predictibility of the VIX-based RSI measure
2. To generate alpha based on the VIX-based RSI measure

**Repository Navigation**
1. ETL Process + Model Training
   1. The scipt is located in ~/pipline/generate_models.py
   2. The ETL config file is located in ~/config/etl_config.json
      1. Within this configuration file, users can specify:
         1. The traded EQ ticker
         2. The VIX (or any other volatility index)
         3. The data download parameter (eg. start and end date, frequency, etc.)
         4. The window for RSI calculation
         5. The benchmark return (%) for the buy signal
         6. The number of lags for feature engineering
         7. The size of the train-test split
   3. The Model Training config file is located in ~/config/model_config.json
      1. Within this configuration file, users can specify:
         1. The name of the target variable to model (buy signal)
         2. The features to be included in the model (based on the lags)
         3. The benchmark probability to be classified as buy
         4. The model name (affects the naming of the model and its associated reports)
2. Model Evaluation
   1. The script is located in ~/pipline/generate_model_evaluation.py
   2. The Model config file is located in ~/config/model_config.json
   3. The stress period evaluation is located in ~/pipline/backtest_stress_period.py
3. Model Interpretation
   1. This project uses SHAP value to explain our machine learning models - please check the documentation here (https://shap.readthedocs.io/en/latest/)
   2. The script is located in ~/pipline/generate_causal_inference.py
4. File Paths
   1. The paths for the various output and config files are located in ~/config/properties.py

**Outputs**
1. The extracted data can be found in ~/data/market_data
2. The pre-trained models can be found in ~/data/models/
3. The model evaluation and backtesting reports can be found in ~/data/model_evaluation/compiled
