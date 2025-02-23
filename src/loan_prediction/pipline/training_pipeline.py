import os, sys

from src.loan_prediction.entity.config_entity import (
    DataIngestionConfig, DataTransformationConfig,
    DataValidationConfig, ModelTrainerConfig,
    TrainingPipelineConfig
)

from src.loan_prediction.entity.artifact_entity import (
    DataIngestionArtifact, DataTransformationArtifact,
    DataValidationArtifact
)

from src.loan_prediction.exception import CustomException


from src.loan_prediction.components.data_ingestion import DataIngestion
from src.loan_prediction.components.data_validation import DataValidation
from src.loan_prediction.components.data_tansformation import DataTransformation
from src.loan_prediction.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
 
        

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
            
            data_validation_artifact=data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(data_transformation_artifact, self.model_trainer_config)

            model_trainer_classification_artifact, model_trainer_regression_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_classification_artifact, model_trainer_regression_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            

            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)    