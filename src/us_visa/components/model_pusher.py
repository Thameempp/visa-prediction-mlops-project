import os
import shutil
import sys

from us_visa.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.exception import UsVisaException
from us_visa.logger import structlog

logging = structlog.get_logger(__name__)


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig = ModelPusherConfig(),
    ):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                raise Exception("Model was not accepted during evaluation.")

            os.makedirs(self.model_pusher_config.saved_model_dir, exist_ok=True)
            destination = os.path.join(
                self.model_pusher_config.saved_model_dir,
                os.path.basename(self.model_evaluation_artifact.trained_model_path),
            )
            shutil.copy2(self.model_evaluation_artifact.trained_model_path, destination)

            return ModelPusherArtifact(saved_model_path=destination)
        except Exception as e:
            raise UsVisaException(e, sys) from e
