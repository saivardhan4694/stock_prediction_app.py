from src.stockpredictor.config.configuration import ConfigurationManager
from src.stockpredictor.components.model_prediction import ModelPrediction
from src.stockpredictor.logging.coustom_log import logger

class ModelPredictionPipeline:
    def __init__(self):
        pass

    def initiate_prediction(self):
        try:
            config = ConfigurationManager()
            model_prediction_config = config.get_model_prediction_config()
            predictor = ModelPrediction(model_prediction_config)
            predictor.forecast_30_days()
        except Exception as e:
            raise e