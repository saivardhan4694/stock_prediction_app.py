from src.stockpredictor.config.configuration import ConfigurationManager
from src.stockpredictor.components.model_trainer import ModelTrainer
from src.stockpredictor.logging.coustom_log import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            config = ConfigurationManager()
            model_training_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(model_training_config)
            model_trainer.train_LSTM_model()
        except Exception as e:
            raise e