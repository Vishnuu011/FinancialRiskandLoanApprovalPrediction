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

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
CLASSIFICATION_MODEL_FILE_NAME = "model.pkl"
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