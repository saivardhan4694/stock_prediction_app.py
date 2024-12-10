from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    raw_stock_data: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    validation_input: Path
    num_of_columns: int
    validation_schema: dict
    validation_report_path: Path
    validation_output_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    transformation_input: Path
    transformation_output: Path

@dataclass
class ModelTrainingConfig:
    root_dir: Path
    training_input: Path
    lstm_model: Path
    test_data_path: Path
    arima_p: int 
    arima_d: int 
    arima_q: int 
    xgboost_n_estimators: int 
    xgboost_learning_rate: float 
    xgboost_max_depth: int 
    xgboost_subsample: float 
    xgboost_colsample_bytree: float 
    xgboost_min_child_weight: int 
    lstm_epochs: int
    lstm_batch_size: int
    lstm_verbose: int

@dataclass
class ModelPredictionConfig:
    root_dir: Path
    model: Path
    test_input: Path
    forecast_input: Path
    predictions_output: Path
    plot_output: Path