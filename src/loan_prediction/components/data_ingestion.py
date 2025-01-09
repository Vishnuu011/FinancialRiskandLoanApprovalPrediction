import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pymongo

from src.loan_prediction.constant import *
from src.loan_prediction.exception import CustomException
from src.loan_prediction.logger import logging
from src.loan_prediction.entity.artifact_entity import DataIngestionArtifact
from src.loan_prediction.entity.config_entity import DataIngestionConfig


from typing import List
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

"""
The DataIngestion class is responsible for:
1. Extracting data from a MongoDB collection and converting it into a pandas DataFrame.
2. Exporting the DataFrame into a feature store as a CSV file.
3. Splitting the data into training and testing datasets and saving them as CSV files.
4. Coordinating the data ingestion pipeline through the `initiate_data_ingestion` method.
 """


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):

        """
        Initialize the DataIngestion class with the given configuration.
        """

        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_collection_as_dataframe(self):

        """
        Read data from MongoDB collection and return it as a pandas DataFrame.
        """

        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]

            df=pd.DataFrame(list(collection.find()))
            logging.info(f"length of data {len(df)}")
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
                
            if "ApplicationDate" in df.columns.to_list():
                df=df.drop(columns=["ApplicationDate"],axis=1)

            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):

        """
        Export the processed DataFrame into a feature store as a CSV file.
        """

        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):

        """
        Split the DataFrame into training and testing datasets and save them as CSV files.
        """

        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_ingestion(self):

        """
        Orchestrate the data ingestion process by calling other methods.
        """
        
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise CustomException(e,sys)
      