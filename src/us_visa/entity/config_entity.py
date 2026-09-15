import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    feature_store_file_path:str = os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    training_file_path:str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
    report_file_path: str = os.path.join(data_validation_dir, "report.yaml")


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
    transformed_train_file_path: str = os.path.join(data_transformation_dir, "transformed", TRAIN_FILE_NAME.replace(".csv", ".npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, "transformed", TEST_FILE_NAME.replace(".csv", ".npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir, "transformed_object", PREPROCSSING_OBJECT_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
    trained_model_file_path: str = os.path.join(model_trainer_dir, "trained_model")
    expected_accuracy: float = 0.6


@dataclass
class ModelPusherConfig:
    saved_model_dir: str = os.path.join("saved_models", training_pipeline_config.timestamp)
