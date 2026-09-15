import sys

from us_visa.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from us_visa.exception import UsVisaException
from us_visa.logger import structlog

logging = structlog.get_logger(__name__)


class ModelEvaluation:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            return ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=self.model_trainer_artifact.test_metric_artifact,
                best_model_path=self.model_trainer_artifact.trained_model_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )
        except Exception as e:
            raise UsVisaException(e, sys) from e
