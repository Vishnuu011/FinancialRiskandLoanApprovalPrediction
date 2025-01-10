import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN

from src.loan_prediction.constant import *
from src.loan_prediction.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.loan_prediction.entity.config_entity import DataTransformationConfig
from src.loan_prediction.exception import CustomException
from src.loan_prediction.logger import logging
from src.loan_prediction.utils.utils import read_yaml_file, save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod    
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)  

    def get_data_transformer_object(self)->Pipeline:

        try:
            od_encoder_col = self._schema_config['categrorical_columns']
            numeric_col = self._schema_config['numerical_columns']
            scaler:StandardScaler=StandardScaler()
            odinalencoder:OrdinalEncoder=OrdinalEncoder()
            scaler_pipeline:Pipeline=Pipeline([("scaler",scaler)])
            odinal_pipeline:Pipeline=Pipeline([("encoder",odinalencoder)])
            preprocessor:ColumnTransformer=ColumnTransformer([
                ("scaler", scaler_pipeline, numeric_col),
                ("od_encoder", odinal_pipeline, od_encoder_col)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_IQR(self, col, df):
        try:
           Q1 = df[col].quantile(0.25)
           Q3 = df[col].quantile(0.75)

           iqr = Q3 - Q1

           upper_limit = Q3 + 1.5 * iqr
           lower_limit = Q1 - 1.5 * iqr

           df.loc[(df[col]>upper_limit),col] = upper_limit
           df.loc[(df[col]<lower_limit),col] = lower_limit

           return df
        except Exception as e:
            raise CustomException(e, sys)             
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            train_df_r=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df_r=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            num_features_out = self._schema_config['outlier_remove_col']

            for features in num_features_out:
                self.remove_outliers_IQR(col=features, df=train_df )

            logging.info("remove_outliers_IQR train_df performed sucessfuly...")

            for features in num_features_out:
                self.remove_outliers_IQR(col=features, df=test_df )

            for features in num_features_out:
                self.remove_outliers_IQR(col=features, df=train_df_r )

            logging.info("remove_outliers_IQR train_df performed sucessfuly...")

            for features in num_features_out:
                self.remove_outliers_IQR(col=features, df=test_df_r )    

            # training dataframe classifiation
            input_feature_train_df=train_df.drop(columns=[CLASSIFICATION_TARGET_COLUMN, REGRESSION_TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[CLASSIFICATION_TARGET_COLUMN]

            # testing dataframe classification
            input_feature_test_df=test_df.drop(columns=[CLASSIFICATION_TARGET_COLUMN, REGRESSION_TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[CLASSIFICATION_TARGET_COLUMN]
        
            # training dataframe regression
            input_feature_train_df_r=train_df.drop(columns=[CLASSIFICATION_TARGET_COLUMN, REGRESSION_TARGET_COLUMN],axis=1)
            target_feature_train_reg_df = train_df[REGRESSION_TARGET_COLUMN]

            # testing dataframe regression
            input_feature_test_df_r=test_df.drop(columns=[CLASSIFICATION_TARGET_COLUMN, REGRESSION_TARGET_COLUMN],axis=1)
            target_feature_test_reg_df = test_df[REGRESSION_TARGET_COLUMN]

            preprocessor=self.get_data_transformer_object()

            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

            preprocessor_object_r=preprocessor.fit(input_feature_train_df_r)
            transformed_input_train_feature_r=preprocessor_object_r.transform(input_feature_train_df_r)
            transformed_input_test_feature_r =preprocessor_object_r.transform(input_feature_test_df_r)

            logging.info(f"shape train preprosser {transformed_input_train_feature.shape}")
            logging.info(f"shape test preprosser {transformed_input_test_feature.shape}")

            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Applying SMOTEENN on Training dataset")

            smt = SMOTEENN(sampling_strategy="minority")

            input_features_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            logging.info("Applied SMOTEENN on training dataset")

            logging.info("Applying SMOTEENN on testing dataset")

            input_features_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )

        
            logging.info("Applied SMOTEENN on testing dataset")

            logging.info("Created train array and test array")

            train_arr = np.c_[input_features_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[ input_features_test_final, np.array(target_feature_test_final) ]

            train_reg_arr = np.c_[transformed_input_train_feature_r, np.array(target_feature_train_reg_df) ]
            test_reg_arr = np.c_[ transformed_input_test_feature_r, np.array(target_feature_test_reg_df) ]

            #save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_train_reg_file_path, array=train_reg_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_reg_file_path, array=test_reg_arr)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_reg_file_path=self.data_transformation_config.transformed_train_reg_file_path,
                transformed_test_reg_file_path=self.data_transformation_config.transformed_test_reg_file_path

            )
            return data_transformation_artifact


        except Exception as e:
            raise CustomException(e, sys)    

