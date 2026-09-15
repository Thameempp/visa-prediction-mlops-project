import sys

import pandas as pd

from us_visa.exception import UsVisaException
from us_visa.utils.main_utils import load_object


class USVisaData:
    def __init__(
        self,
        continent: str,
        education_of_employee: str,
        has_job_experience: str,
        requires_job_training: str,
        no_of_employees: int,
        yr_of_estab: int,
        region_of_employment: str,
        prevailing_wage: float,
        unit_of_wage: str,
        full_time_position: str,
    ):
        self.continent = continent
        self.education_of_employee = education_of_employee
        self.has_job_experience = has_job_experience
        self.requires_job_training = requires_job_training
        self.no_of_employees = no_of_employees
        self.yr_of_estab = yr_of_estab
        self.region_of_employment = region_of_employment
        self.prevailing_wage = prevailing_wage
        self.unit_of_wage = unit_of_wage
        self.full_time_position = full_time_position

    def get_usvisa_input_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame([self.__dict__])
        except Exception as e:
            raise UsVisaException(e, sys) from e


class USVisaClassifier:
    def __init__(self, preprocessing_object_path: str, trained_model_path: str):
        self.preprocessing_object_path = preprocessing_object_path
        self.trained_model_path = trained_model_path

    def predict(self, dataframe: pd.DataFrame):
        try:
            preprocessing_obj = load_object(self.preprocessing_object_path)
            trained_model = load_object(self.trained_model_path)
            transformed_data = preprocessing_obj.transform(dataframe)
            return trained_model.predict(transformed_data)
        except Exception as e:
            raise UsVisaException(e, sys) from e
