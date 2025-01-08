import os
import sys
import json
import certifi

import pandas as pd
import numpy as np
import pymongo
from src.loan_prediction.exception import CustomException
from src.loan_prediction.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONGO_URL =os.getenv("MONGO_DB_URL")


ca = certifi.where()


class FinancialRiskExtart():
    def __init__(self):

        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def csv_to_json(self, file_path):

        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e, sys)
        
    def insert_data_mongodb(self, records, database, colloction):

        try:
            self.database = database
            self.colloction = colloction
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_URL)
            self.database = self.mongo_client[self.database]

            self.colloction = self.database[self.colloction]
            self.colloction.insert_many(self.records)
            return (len(self.records))

        except Exception as e:
            raise CustomException(e, sys)   


if __name__=='__main__':

    FILE_PATH = "FinancialRiskData\Loan.csv"
    DATABASE="VISHNUAIDATA"
    Collection="FinancialRisk"
    FinancialRiskobj=FinancialRiskExtart()
    records=FinancialRiskobj.csv_to_json(file_path=FILE_PATH)
    print(records)
    no_of_records=FinancialRiskobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)
        





