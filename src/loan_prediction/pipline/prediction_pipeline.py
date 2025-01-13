import os,sys
import pandas as pd
import sklearn
import numpy as np

from src.loan_prediction.exception import CustomException
from src.loan_prediction.logger import logging
from src.loan_prediction.utils.utils import load_object
from src.loan_prediction.utils.ml_utils.estimator import FinacalRiskModel



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
                "Age": [self.Age],
                "AnnualIncome": [self.AnnualIncome],
                "CreditScore": [self.CreditScore],
                "EmploymentStatus": [self.EmploymentStatus],
                "EducationLevel": [self.EducationLevel],
                "Experience": [self.Experience],
                "LoanAmount": [self.LoanAmount],
                "LoanDuration": [self.LoanDuration],
                "MaritalStatus": [self.MaritalStatus],
                "NumberOfDependents": [self.NumberOfDependents],
                "HomeOwnershipStatus": [self.HomeOwnershipStatus],
                "MonthlyDebtPayments": [self.MonthlyDebtPayments],
                "CreditCardUtilizationRate": [self.CreditCardUtilizationRate],
                "NumberOfOpenCreditLines": [self.NumberOfOpenCreditLines],
                "NumberOfCreditInquiries": [self.NumberOfCreditInquiries],
                "DebtToIncomeRatio": [self.DebtToIncomeRatio],
                "BankruptcyHistory": [self.BankruptcyHistory],
                "LoanPurpose": [self.LoanPurpose],
                "PreviousLoanDefaults": [self.PreviousLoanDefaults],
                "PaymentHistory": [self.PaymentHistory],
                "LengthOfCreditHistory": [self.LengthOfCreditHistory],
                "SavingsAccountBalance": [self.SavingsAccountBalance],
                "CheckingAccountBalance": [self.CheckingAccountBalance],
                "TotalAssets": [self.TotalAssets],
                "TotalLiabilities": [self.TotalLiabilities],
                "MonthlyIncome": [self.MonthlyIncome],
                "UtilityBillsPaymentHistory": [self.UtilityBillsPaymentHistory],
                "JobTenure": [self.JobTenure],
                "NetWorth": [self.NetWorth],
                "BaseInterestRate": [self.BaseInterestRate],
                "InterestRate": [self.InterestRate],
                "MonthlyLoanPayment": [self.MonthlyLoanPayment],
                "TotalDebtToIncomeRatio": [self.TotalDebtToIncomeRatio],
                
            }

            return pd.DataFrame(custom_data_input_dict, index=[0])

        except Exception as e:
            raise CustomException(e, sys)         