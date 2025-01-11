from src.loan_prediction.logger import logging
from src.loan_prediction.exception import CustomException
from src.loan_prediction.utils.utils import load_object

import numpy
print(numpy.__version__)



model = load_object("final_model/model_regression.pkl")
model_classi = load_object("final_model/model_classification.pkl")
print(model)
print(model_classi)

