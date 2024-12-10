from src.stockpredictor.config.configuration import ConfigurationManager
from src.stockpredictor.components.data_ingestion import DataIngestion
from src.stockpredictor.logging.coustom_log import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config, stock_name= "NVDA", start_date="2013-01-01")
            data_ingestion.get_data()
        except Exception as e:
            raise e