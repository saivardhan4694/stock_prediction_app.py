from src.stockpredictor.config.configuration import ConfigurationManager
from src.stockpredictor.components.data_transformation import DataTransformation
from src.stockpredictor.logging.coustom_log import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(data_transformation_config)
            data_transformation.transform_data()
        except Exception as e:
            raise e