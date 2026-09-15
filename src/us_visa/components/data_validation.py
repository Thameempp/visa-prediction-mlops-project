import os
import sys
from typing import List

import pandas as pd

from us_visa.constants import SCHEMA_FILE_PATH
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.exception import UsVisaException
from us_visa.logger import structlog
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file

logging = structlog.get_logger(__name__)


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema_config["columns"]
            return len(dataframe.columns) == len(expected_columns)
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def validate_required_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns: List[str] = list(self.schema_config["columns"].keys())
            missing_columns = [column for column in expected_columns if column not in dataframe.columns]
            if missing_columns:
                logging.error("Missing columns", missing_columns=missing_columns)
                return False
            return True
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            validation_status = all(
                [
                    self.validate_number_of_columns(train_df),
                    self.validate_number_of_columns(test_df),
                    self.validate_required_columns(train_df),
                    self.validate_required_columns(test_df),
                ]
            )

            report = {"validation_status": validation_status}
            os.makedirs(os.path.dirname(self.data_validation_config.report_file_path), exist_ok=True)
            write_yaml_file(self.data_validation_config.report_file_path, report, replace=True)

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                report_file_path=self.data_validation_config.report_file_path,
            )
        except Exception as e:
            raise UsVisaException(e, sys) from e
