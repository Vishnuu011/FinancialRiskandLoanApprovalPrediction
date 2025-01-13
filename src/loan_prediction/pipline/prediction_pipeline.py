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



class CustomData:
    def __init__(  self,
        Age: int,
        AnnualIncome: int,
        CreditScore:int,
        EmploymentStatus: str,
        EducationLevel: str,
        Experience: int,
        LoanAmount: int,
        LoanDuration: int,
        MaritalStatus:str,
        NumberOfDependents:int,
        HomeOwnershipStatus:str,
        MonthlyDebtPayments:int,
        CreditCardUtilizationRate:float,
        NumberOfOpenCreditLines:int,
        NumberOfCreditInquiries:int,
        DebtToIncomeRatio:float,
        BankruptcyHistory:int,
        LoanPurpose:str,
        PreviousLoanDefaults:int,
        PaymentHistory:int,
        LengthOfCreditHistory:int,
        SavingsAccountBalance:int,
        CheckingAccountBalance:int,
        TotalAssets:int,
        TotalLiabilities:int,
        MonthlyIncome:float,
        UtilityBillsPaymentHistory:float,
        JobTenure:int,
        NetWorth:int,
        BaseInterestRate:float,
        InterestRate:float,
        MonthlyLoanPayment:float,
        TotalDebtToIncomeRatio:float):

        self.Age = Age
        self.AnnualIncome = AnnualIncome
        self.CreditScore = CreditScore
        self.EmploymentStatus =EmploymentStatus
        self.EducationLevel = EducationLevel
        self.Experience = Experience
        self.LoanAmount = LoanAmount
        self.LoanDuration = LoanDuration
        self.MaritalStatus =MaritalStatus
        self.NumberOfDependents = NumberOfDependents
        self.HomeOwnershipStatus = HomeOwnershipStatus
        self.MonthlyDebtPayments = MonthlyDebtPayments
        self.CreditCardUtilizationRate = CreditCardUtilizationRate
        self.NumberOfOpenCreditLines =NumberOfOpenCreditLines
        self.NumberOfCreditInquiries = NumberOfCreditInquiries
        self.DebtToIncomeRatio = DebtToIncomeRatio
        self.BankruptcyHistory = BankruptcyHistory
        self.LoanPurpose = LoanPurpose
        self.PreviousLoanDefaults =PreviousLoanDefaults
        self.PaymentHistory = PaymentHistory
        self.LengthOfCreditHistory = LengthOfCreditHistory
        self.SavingsAccountBalance = SavingsAccountBalance
        self.CheckingAccountBalance = CheckingAccountBalance
        self.TotalAssets =TotalAssets
        self.TotalLiabilities = TotalLiabilities
        self.MonthlyIncome = MonthlyIncome
        self.UtilityBillsPaymentHistory = UtilityBillsPaymentHistory
        self.JobTenure =JobTenure
        self.NetWorth = NetWorth
        self.BaseInterestRate = BaseInterestRate
        self.InterestRate = InterestRate
        self.MonthlyLoanPayment = MonthlyLoanPayment
        self.TotalDebtToIncomeRatio =TotalDebtToIncomeRatio

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.Age],
                "InternetService": [self.AnnualIncome],
                "Contract": [self.CreditScore],
                "tenure": [self.EmploymentStatus],
                "MonthlyCharges": [self.EducationLevel],
                "gender": [self.Experience],
                "InternetService": [self.LoanAmount],
                "Contract": [self.LoanDuration],
                "tenure": [self.MaritalStatus],
                "MonthlyCharges": [self.NumberOfDependents],
                "gender": [self.HomeOwnershipStatus],
                "InternetService": [self.MonthlyDebtPayments],
                "Contract": [self.CreditCardUtilizationRate],
                "tenure": [self.NumberOfOpenCreditLines],
                "MonthlyCharges": [self.NumberOfCreditInquiries],
                "gender": [self.DebtToIncomeRatio],
                "InternetService": [self.BankruptcyHistory],
                "Contract": [self.LoanPurpose],
                "tenure": [self.PreviousLoanDefaults],
                "MonthlyCharges": [self.PaymentHistory],
                "gender": [self.LengthOfCreditHistory],
                "InternetService": [self.SavingsAccountBalance],
                "Contract": [self.CheckingAccountBalance],
                "tenure": [self.TotalAssets],
                "MonthlyCharges": [self.TotalLiabilities],
                "gender": [self.MonthlyIncome],
                "InternetService": [self.UtilityBillsPaymentHistory],
                "Contract": [self.JobTenure],
                "tenure": [self.NetWorth],
                "MonthlyCharges": [self.BaseInterestRate],
                "Contract": [self.InterestRate],
                "tenure": [self.MonthlyLoanPayment],
                "MonthlyCharges": [self.TotalDebtToIncomeRatio],
                
            }

            return pd.DataFrame(custom_data_input_dict, index=[0])

        except Exception as e:
            raise CustomException(e, sys)         