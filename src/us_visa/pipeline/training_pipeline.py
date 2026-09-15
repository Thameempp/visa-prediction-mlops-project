import sys
from us_visa.exception import UsVisaException
from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.data_validation import DataValidation
from us_visa.components.model_evaluation import ModelEvaluation
from us_visa.components.model_pusher import ModelPusher
from us_visa.components.model_trainer import ModelTrainer



from us_visa.entity.config_entity import (DataIngestionConfig,
                                          )

from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataTransformationArtifact,
                                            DataValidationArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact,
                                            ModelTrainerArtifact,
                                            )


from us_visa.logger import structlog

logging = structlog.get_logger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(model_trainer_artifact=model_trainer_artifact)
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact)
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact)
            self.start_model_pusher(model_evaluation_artifact)

            print("Pipeline completed successfully!")

        except Exception as e:
            raise UsVisaException(e, sys)
