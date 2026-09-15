import os
import sys

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from us_visa.constants import MODEL_FILE_NAME
from us_visa.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.exception import UsVisaException
from us_visa.logger import structlog
from us_visa.utils.main_utils import load_numpy_array_data, save_object

logging = structlog.get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def train_model(self, x_train, y_train):
        try:
            model = KNeighborsClassifier()
            model.fit(x_train, y_train)
            return model
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            model = self.train_model(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            if test_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model accuracy {test_accuracy} is below expected "
                    f"{self.model_trainer_config.expected_accuracy}"
                )

            model_file_path = os.path.join(self.model_trainer_config.trained_model_file_path, MODEL_FILE_NAME)
            save_object(model_file_path, model)

            return ModelTrainerArtifact(
                trained_model_file_path=model_file_path,
                train_metric_artifact=train_accuracy,
                test_metric_artifact=test_accuracy,
            )
        except Exception as e:
            raise UsVisaException(e, sys) from e
