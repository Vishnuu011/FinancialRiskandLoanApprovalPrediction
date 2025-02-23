from src.loan_prediction.logger import logging
from src.loan_prediction.exception import CustomException
from src.loan_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.loan_prediction.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.loan_prediction.components.data_ingestion import DataIngestion
from src.loan_prediction.components.data_validation import DataValidation
from src.loan_prediction.components.data_tansformation import DataTransformation
from src.loan_prediction.components.model_trainer import ModelTrainer
from loan_prediction.pipline.training_pipeline import TrainingPipeline



import sys

if __name__=='__main__':
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        raise CustomException(e, sys)  