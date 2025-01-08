from src.loan_prediction.logger import logging
from src.loan_prediction.exception import CustomException
import sys


try:
    logging.info("Enter the try block")
    a=1/0
    print("This will not be printed",a)
except Exception as e:
     raise CustomException(e,sys)