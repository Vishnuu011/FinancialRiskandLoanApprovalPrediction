from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import sys
from src.loan_prediction.exception import CustomException
from src.loan_prediction.logger import logging
from src.loan_prediction.entity.artifact_entity import ClassificationMetricArtifact, RegressionMetricArtifact

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:

        model_accuracy = accuracy_score(y_true, y_pred)    
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score,
                    accuracy=model_accuracy)
        return classification_metric
    except Exception as e:
        raise CustomException(e,sys)
    

def get_Regression_score(y_true,y_pred)->RegressionMetricArtifact:
    try:
            
        model_rmse = root_mean_squared_error(y_true, y_pred)
        model_mae = mean_absolute_error(y_true, y_pred)
        model_mse=mean_squared_error(y_true,y_pred)
        model_r2score = r2_score(y_true,y_pred)

        Regression_metric =  RegressionMetricArtifact(Rmse=model_rmse, mae=model_mae, mse=model_mse,
                                                          r2score=model_r2score)
        return Regression_metric
    except Exception as e:
        raise CustomException(e,sys)    