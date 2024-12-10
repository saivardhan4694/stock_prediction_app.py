import os
from pathlib import Path
import yfinance as yf
from src.stockpredictor.utils.common import save_as_csv
from src.stockpredictor.logging.coustom_log import logger
from src.stockpredictor.entity import DataIngestionConfig
import pandas as pd
from datetime import datetime

class DataIngestion:
    def __init__(self, config: DataIngestionConfig,
                 stock_name: str,
                 start_date: str):
        self.config = config
        self.stock_name = stock_name
        self.start_date = start_date
        

    def get_data(self):
        try:
            stock_dataframe = yf.download(self.stock_name, start=self.start_date, end= datetime.today().strftime('%Y-%m-%d'), group_by='ticker')
            if isinstance(stock_dataframe.columns, pd.MultiIndex):
                stock_dataframe = stock_dataframe[self.stock_name] 
            stock_dataframe.reset_index(inplace=True)
            save_as_csv(dataframe = stock_dataframe, path_to_csv = Path(self.config.raw_stock_data))
            logger.info(f"Data Ingested for {self.stock_name}")
        except Exception as e:
            raise e
