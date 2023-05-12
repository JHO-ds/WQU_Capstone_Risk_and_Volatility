import yfinance as yf
import pandas as pd
import numpy as np
import os
import math

import config.properties as p
import src.utils.utils as utils


class Market_Data_Loader:

    def __init__(self):
        # import the settings
        etl_config = utils.read_json(p.etl_config_path)

        # initialize the settings
        self.data_loading_config = etl_config.get("data_loading")
        self.data_processing_config = etl_config.get("data_processing")
        self.train_test_config = etl_config.get("train_test_split")

        self.vix_ticker = self.data_loading_config.get("VIX_Ticker_Name", "^VIX")
        self.eq_ticker = self.data_loading_config.get("EQ_Ticker_Name", "VOO")
        self.all_tickers = " ".join([self.vix_ticker, self.eq_ticker])
        self.start_date = self.data_loading_config.get("start_date")
        self.end_date = self.data_loading_config.get("end_date")
        self.interval = self.data_loading_config.get("download_interval")
        self.col_name = self.data_loading_config.get("col_name")

        self.window = self.data_processing_config.get("window_period")
        self.benchmark = self.data_processing_config.get("target_benchmark")
        self.lag = self.data_processing_config.get("lags", 7)

        self.train_size = self.train_test_config.get("train_size")

    def market_data_download(self) -> pd.DataFrame:
        return yf.download(tickers=self.all_tickers,
                           start=self.start_date,
                           end=self.end_date,
                           interval=self.interval)[self.col_name].dropna()

    def market_data_generate_returns(self, market_data: pd.DataFrame) -> pd:
        return np.log(market_data).diff() * 100

    def generate_features(self, market_data: pd.DataFrame) -> pd:
        # generate RSI
        vix_rsi = market_data.copy(deep=True)
        vix_rsi["Gain"] = vix_rsi[self.vix_ticker].apply(lambda x: x if x > 0 else 0)
        vix_rsi["Loss"] = vix_rsi[self.vix_ticker].apply(lambda x: -x if x < 0 else 0)
        vix_rsi["Avg_Gain"] = vix_rsi["Gain"].rolling(self.window).apply(lambda x: x[x != 0].mean())
        vix_rsi["Avg_Loss"] = vix_rsi["Loss"].rolling(self.window).apply(lambda x: x[x != 0].mean())
        vix_rsi["RSI"] = vix_rsi["Avg_Gain"] / vix_rsi["Avg_Loss"]
        vix_rsi["RSI"] = vix_rsi["RSI"].shift(1)

        # generate the lags
        for i in range(1, self.lag+1):
            vix_rsi[f"lag_{i}"] = vix_rsi[self.vix_ticker].shift(i)

        # generate binary variable
        vix_rsi["target"] = vix_rsi[self.eq_ticker].apply(lambda x: int(x > self.benchmark))

        # clear intermediary columns
        processed_market_data = vix_rsi.copy(deep=True). \
            drop(columns=["Gain", "Loss", "Avg_Gain", "Avg_Loss"]). \
            dropna()

        return processed_market_data

    def export_market_data(self, market_data: pd.DataFrame):
        file_name = f"{self.eq_ticker}_{self.start_date}_{self.end_date}.csv"
        market_data.to_csv(os.path.join(p.market_data_output_folder, file_name), index=False)

    def train_test_split(self, market_data: pd.DataFrame):
        mkt_env_size = market_data.shape[0]
        train_index = math.floor(mkt_env_size * self.train_size)
        train_df = market_data.iloc[:train_index]
        test_df = market_data.iloc[train_index:]
        return train_df, test_df

    def etl_process(self):
        market_data = self.market_data_download()
        log_returns = self.market_data_generate_returns(market_data)
        processed_market_data = self.generate_features(log_returns)
        self.export_market_data(processed_market_data)
        train_df, test_df = self.train_test_split(processed_market_data)
        train_df.to_csv(os.path.join(p.market_data_output_folder, "train_set.csv"))
        test_df.to_csv(os.path.join(p.market_data_output_folder, "test_set.csv"))
