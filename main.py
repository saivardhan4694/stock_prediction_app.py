from src.stockpredictor.logging.coustom_log import logger
from src.stockpredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.stockpredictor.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.stockpredictor.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.stockpredictor.pipeline.stage_04_model_training import ModelTrainingPipeline
from src.stockpredictor.pipeline.stage_05_prediction_pipeline import ModelPredictionPipeline

# STAGE_NAME = "Data Ingestion Stage"
# try:
#     logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.initiate_data_ingestion()
#     logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\n=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Data Validation Stage"
# try:
#     logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
#     data_validation = DataValidationTrainingPipeline()
#     data_validation.initiate_data_validation()
#     logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\n=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Data Transforamtion Stage"
# try:
#     logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
#     data_transformation = DataTransformationTrainingPipeline()
#     data_transformation.initiate_data_transformation()
#     logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\n=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "MOdel training Stage"
# try:
#     logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
#     model_training = ModelTrainingPipeline()
#     model_training.initiate_model_training()
#     logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\n=================x")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "MOdel prediction Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    model_prediction = ModelPredictionPipeline()
    model_prediction.initiate_prediction()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\n=================x")
except Exception as e:
    logger.exception(e)
    raise e