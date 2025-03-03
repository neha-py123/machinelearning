import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pickle

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function returns a preprocessing pipeline that applies transformations.
        """
        try:
            logging.info("Creating preprocessing pipelines...")

            # Define numerical and categorical columns (Update based on your dataset)
            numerical_columns = ["writing_score", "reading_score"]  # Example numeric columns
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education","lunch","test_preparation_course"]  # Example categorical columns

            # Pipelines for transformation
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Handling missing values
                ("scaler", StandardScaler())  # Scaling numerical values
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Handling missing categorical values
                ("encoder", OneHotEncoder(handle_unknown="ignore"))  # Encoding categorical values
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train & test data, applies transformation, and saves the processed data.
        """
        try:
            logging.info("Loading train & test data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Creating preprocessing object...")
            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"  # Update based on your dataset
            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_test = test_df[target_column]

            logging.info("Applying transformations...")
            input_features_train = preprocessing_obj.fit_transform(input_features_train)
            input_features_test = preprocessing_obj.transform(input_features_test)

            # Save preprocessor object
            with open(self.transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessing_obj, f)

            logging.info("Transformation completed.")

            return input_features_train, target_train, input_features_test, target_test, self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
