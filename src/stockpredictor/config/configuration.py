from src.stockpredictor.utils.common import create_directories
from src.stockpredictor.constants.__init import *
from src.stockpredictor.entity import (DataIngestionConfig,
                                       DataValidationConfig,
                                       DataTransformationConfig,
                                       ModelTrainingConfig)
from src.stockpredictor.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH) -> None:
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            raw_stock_data= config.raw_stock_data
        )
        return data_ingestion_config
    
    def get_data_validation_config(self):
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            validation_input= config.validation_input,
            num_of_columns= config.num_of_columns,
            validation_schema = config.validation_schema,
            validation_report_path = config.validation_report_path,
            validation_output_path= config.validation_output_path
        )

        return data_validation_config
    
    def get_data_transformation_config(self):
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir= config.root_dir,
            transformation_input = config.transformation_input,
            transformation_output = config.transformation_output
        )

        return data_transformation_config
    
    def get_model_trainer_config(self):
        config = self.config.model_trainer
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            training_input = config.training_input,
            lstm_model = config.lstm_model,
            arima_p = params.arima.p,
            arima_d = params.arima.d ,
            arima_q = params.arima.q,
            xgboost_n_estimators = params.xgboost.n_estimators,
            xgboost_learning_rate = params.xgboost.learning_rate ,
            xgboost_max_depth = params.xgboost.max_depth,
            xgboost_subsample = params.xgboost.subsample ,
            xgboost_colsample_bytree = params.xgboost.colsample_bytree ,
            xgboost_min_child_weight = params.xgboost.min_child_weight,
            lstm_batch_size=params.LSTM.batch_size,
            lstm_epochs=params.LSTM.epochs,
            lstm_verbose=params.LSTM.verbose
        )

        return model_trainer_config