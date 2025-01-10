import os
import sys
import numpy as np




CLASSIFICATION_TARGET_COLUMN = "LoanApproved"
REGRESSION_TARGET_COLUMN = "RiskScore"
PIPELINE_NAME: str = "loan_prediction"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "Loan.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
TRAIN_REGRESSOR_FILE_NAME: str = "train_reg.csv"
TEST_REGRESSOR_FILE_NAME: str = "test_reg.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
CLASSIFICATION_MODEL_FILE_NAME = "model_classification.pkl"
REGRESSION_MODEL_FILE_NAME = "model_regression.pkl"



"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "FinancialRisk"
DATA_INGESTION_DATABASE_NAME: str = "VISHNUAIDATA"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

DATA_TRANSFORMATION_TRAIN_REG_FILE_PATH: str = "train_reg.npy"
DATA_TRANSFORMATION_TEST_REG_FILE_PATH: str = "test_reg.npy"


"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_CLASSIFI_MODEL_NAME: str = "model_classification.pkl"
MODEL_TRAINER_TRAINED_REGRESS_MODEL_NAME: str = "model_regression.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.9
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

TRAINING_BUCKET_NAME = "FinancialRisk"