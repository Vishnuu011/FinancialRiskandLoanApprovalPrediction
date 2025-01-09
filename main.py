from src.loan_prediction.logger import logging
from src.loan_prediction.exception import CustomException
from src.loan_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.loan_prediction.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig
from src.loan_prediction.components.data_ingestion import DataIngestion
from src.loan_prediction.components.data_validation import DataValidation


import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)
    except Exception as e:
        raise CustomException(e, sys)  