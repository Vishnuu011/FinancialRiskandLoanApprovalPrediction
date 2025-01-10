from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str  
    transformed_train_reg_file_path: str
    transformed_test_reg_file_path: str   


@dataclass
class ClassificationMetricArtifact:
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class RegressionMetricArtifact:
    Rmse: float
    mae: float
    mse: float
    r2score: float   
    
@dataclass
class ModelTrainerArtifact:
    trained_model_classification_file_path: str
    trained_model_regression_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact  
    train_metric_artifact_r: RegressionMetricArtifact
    test_metric_artifact_r: RegressionMetricArtifact    