from src.loan_prediction.logger import logging
from src.loan_prediction.exception import CustomException
from src.loan_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.loan_prediction.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.loan_prediction.components.data_ingestion import DataIngestion
from src.loan_prediction.components.data_validation import DataValidation
from src.loan_prediction.components.data_tansformation import DataTransformation
from src.loan_prediction.components.model_trainer import ModelTrainer



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
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")
        logging.info("Model Training sstared")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(data_transformation_artifact,model_trainer_config)
        model_trainer_classification_artifact, model_trainer_regression_artifact=model_trainer.initiate_model_trainer()

        print(model_trainer_classification_artifact)
        print(model_trainer_regression_artifact)

        logging.info("Model Training artifact created")
    except Exception as e:
        raise CustomException(e, sys)  