import os, sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

from src.loan_prediction.exception import CustomException
from src.loan_prediction.logger import logging
from src.loan_prediction.entity.artifact_entity import ModelTrainerClassificationArtifact, ModelTrainerRegressionArtifact, DataTransformationArtifact
from src.loan_prediction.entity.config_entity import ModelTrainerConfig
from src.loan_prediction.utils.utils import (
    save_object, load_object,
    load_numpy_array_data, evaluate_models_classification,
    evaluate_models_regression
)
from src.loan_prediction.utils.ml_utils.metrics import get_classification_score, get_Regression_score
from src.loan_prediction.utils.ml_utils.estimator import FinacalRiskModel


class ModelTrainer:
    def __init__(self, data_trasformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        
        try:
            self.data_transformation_artifact=data_trasformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
            }
        #classifiction hyperparametes
        params={
            "Decision Tree":{
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2]
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            "Gradient Boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2]
            },
            "AdaBoost":{
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            }
            
        }
        #regression hyperparametes
        model_report_cl:dict=evaluate_models_classification(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        print(model_report_cl)
        ## To get best model classification score from dict
        best_model_score = max(sorted(model_report_cl.values()))

        ## To get best model name from dict classification

        best_model_name = list(model_report_cl.keys())[
            list(model_report_cl.values()).index(best_model_score)
        ]
        print(best_model_name)


        #classification metrics
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        

        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_classification_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        fian_Model_classi=FinacalRiskModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_classification_model_file_path,obj=FinacalRiskModel)

        #model pusher classification
        save_object("final_model/model_classification.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerClassificationArtifact(trained_model_classification_file_path=self.model_trainer_config.trained_classification_model_file_path,
                                                    train_metric_artifact=classification_train_metric,
                                                    test_metric_artifact=classification_test_metric)
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


    def train_model_regression(self,X_train,y_train,x_test,y_test):
      
        models_reg = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
            }
        #regression hyperparametes
        params_reg = {
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # Suitable for regression
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['auto', 'sqrt', 'log2'],  # Feature selection for regression
                'bootstrap': [True, False]
            },
            "Gradient Boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            "AdaBoost": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'loss': ['linear', 'square', 'exponential']  # Suitable for regression
            }
        }
        model_report_reg:dict=evaluate_models_regression(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models_reg,param=params_reg)
        
        print(model_report_reg)

        ## To get best model regression score from dict 
        best_model_score_reg = max(sorted(model_report_reg.values()))

        ## To get best model name from dict regression

        best_model_name_reg = list(model_report_reg.keys())[
            list(model_report_reg.values()).index(best_model_score_reg)
        ]
        print(best_model_name_reg)
        #regression 
        best_model_reg = models_reg[best_model_name_reg]
        y_train_pred=best_model_reg.predict(X_train)

        regression_train_metric=get_Regression_score(y_true=y_train,y_pred=y_train_pred)
        

        y_test_pred=best_model_reg.predict(x_test)
        regression_test_metric=get_Regression_score(y_true=y_test,y_pred=y_test_pred)


        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        #regrssion
        model_dir_path_r = os.path.dirname(self.model_trainer_config.trained_regression_model_file_path)
        os.makedirs(model_dir_path_r,exist_ok=True)

        fian_Model_reg=FinacalRiskModel(preprocessor=preprocessor,model=best_model_reg)
        save_object(self.model_trainer_config.trained_regression_model_file_path,obj=FinacalRiskModel)
        #model pusher regression
        save_object("final_model/model_regression.pkl",best_model_reg)
        

        ## Model Trainer Artifact
        model_trainer_artifact_r=ModelTrainerRegressionArtifact(trained_model_regression_file_path=self.model_trainer_config.trained_regression_model_file_path,
                                                    train_metric_artifact_r=regression_train_metric,
                                                    test_metric_artifact_r=regression_test_metric)
        logging.info(f"Model trainer artifact: {model_trainer_artifact_r}")
        return model_trainer_artifact_r    
     

    def initiate_model_trainer(self)->tuple:

        try:
            train_file_path_classification =self.data_transformation_artifact.transformed_train_file_path
            test_file_path_classification = self.data_transformation_artifact.transformed_test_file_path

            train_file_path_regression = self.data_transformation_artifact.transformed_train_reg_file_path
            test_file_path_regression = self.data_transformation_artifact.transformed_test_reg_file_path

            train_arr_cl = load_numpy_array_data(train_file_path_classification)
            test_arr_cl = load_numpy_array_data(test_file_path_classification)
            
            train_arr_reg = load_numpy_array_data(train_file_path_regression)
            test_arr_reg = load_numpy_array_data(test_file_path_regression)
            
            #classification train and test split also drop both target columns
            x_train, y_train, x_test, y_test = (
                train_arr_cl[:, :-1],
                train_arr_cl[:, -1],
                test_arr_cl[:, :-1],
                test_arr_cl[:, -1],
            )

            #regression train and test split also drop both target columns 
            train_x, train_y, test_x, test_y = (
                train_arr_reg[:, :-1],
                train_arr_reg[:, -1],
                test_arr_reg[:, :-1],
                test_arr_reg[:, -1],
            )
            model_trainer_classification_artifact=self.train_model(x_train,y_train,x_test,y_test)
            print("____________________________________________________________________________")

            print("======================Regresssion triner started============================")
            print("_____________________________________________________________________________")
            model_trainer_regression_artifact=self.train_model_regression(train_x, train_y, test_x, test_y )

            return model_trainer_classification_artifact, model_trainer_regression_artifact
        except Exception as e:
            raise CustomException(e, sys)      