from src.stockpredictor.config.configuration import ConfigurationManager
from src.stockpredictor.components.data_validation import DataValidation
from src.stockpredictor.logging.coustom_log import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(data_validation_config)
            data_validation.validate_data()
        except Exception as e:
            raise e