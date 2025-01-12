import os,sys
import pandas as pd
import sklearn
import numpy as np

from src.loan_prediction.exception import CustomException
from src.loan_prediction.logger import logging
from src.loan_prediction.utils.utils import load_object
from src.loan_prediction.utils.ml_utils.estimator import FinacalRiskModel

class PredictionPipeline:
    def __init__(self):

        pass

    def prediction_classification(self, feature):

        try:
            model_path_classification_path=os.path.join("final_model","model_classification.pkl")
            preprocessor_path=os.path.join('final_model','preprocessor.pkl')
            model_classification = load_object(file_path=model_path_classification_path)
            preprossers = load_object(file_path=preprocessor_path)

            scaled_data = preprossers.transform(feature)
            preds = model_classification.predict(scaled_data)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
    def prediction_regression(self, feature):

        try:
            model_path_regression_path=os.path.join("final_model","model_regression.pkl")
            preprocessor_path=os.path.join('final_model','preprocessor.pkl')
            model_regression = load_object(file_path=model_path_regression_path)
            preprossers = load_object(file_path=preprocessor_path)

            scaled_data = preprossers.transform(feature)
            preds = model_regression.predict(scaled_data)
            return preds
        except Exception as e:
            raise CustomException(e, sys)    