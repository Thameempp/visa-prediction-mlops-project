import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from us_visa.constants import CURRENT_YEAR, TARGET_COLUMN
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.exception import UsVisaException
from us_visa.logger import structlog
from us_visa.utils.main_utils import save_numpy_array_data, save_object

logging = structlog.get_logger(__name__)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise UsVisaException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise UsVisaException(e, sys) from e

    @staticmethod
    def feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            dataframe = dataframe.copy()
            dataframe["company_age"] = CURRENT_YEAR - dataframe["yr_of_estab"]
            return dataframe.drop(columns=["case_id", "yr_of_estab"], errors="ignore")
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            one_hot_columns = ["continent", "unit_of_wage", "region_of_employment"]
            ordinal_columns = [
                "has_job_experience",
                "requires_job_training",
                "full_time_position",
                "education_of_employee",
            ]
            power_transform_columns = ["no_of_employees", "company_age"]
            numeric_columns = ["no_of_employees", "prevailing_wage", "company_age"]

            return ColumnTransformer(
                [
                    ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"), one_hot_columns),
                    (
                        "OrdinalEncoder",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ordinal_columns,
                    ),
                    (
                        "PowerTransformer",
                        Pipeline(steps=[("transformer", PowerTransformer(method="yeo-johnson"))]),
                        power_transform_columns,
                    ),
                    ("StandardScaler", StandardScaler(), numeric_columns),
                ],
                remainder="drop",
            )
        except Exception as e:
            raise UsVisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception("Data validation failed. Transformation cannot continue.")

            train_df = self.feature_engineering(self.read_data(self.data_validation_artifact.valid_train_file_path))
            test_df = self.feature_engineering(self.read_data(self.data_validation_artifact.valid_test_file_path))

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].replace({"Certified": 1, "Denied": 0})

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].replace({"Certified": 1, "Denied": 0})

            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            smote = SMOTEENN(random_state=42, sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smote.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessing_obj)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise UsVisaException(e, sys) from e
