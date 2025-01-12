from flask import Flask,request,render_template, url_for
from src.loan_prediction.pipline.prediction_pipeline import CustomData
from src.loan_prediction.utils.ml_utils.estimator import FinacalRiskModel
from src.loan_prediction.utils.utils import load_object



app=Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("index.html")
    else:
        data=CustomData(
            Age=int(request.form.get("Age")),
            AnnualIncome=int(request.form.get("AnnualIncome")),
            CreditScore=int(request.form.get("CreditScore")),
            EmploymentStatus=request.form.get("EmploymentStatus"),
            EducationLevel=request.form.get("EducationLevel"),
            Experience=int(request.form.get("Experience")),
            LoanAmount=int(request.form.get("LoanAmount")),
            LoanDuration=int(request.form.get("LoanDuration")),
            MaritalStatus=request.form.get("MaritalStatus"),
            NumberOfDependents=int(request.form.get("NumberOfDependents")),
            HomeOwnershipStatus=request.form.get("HomeOwnershipStatus"),
            MonthlyDebtPayments=int(request.form.get("MonthlyDebtPayments")),
            CreditCardUtilizationRate=float(request.form.get("CreditCardUtilizationRate")),
            NumberOfOpenCreditLines=int(request.form.get("NumberOfOpenCreditLines")),
            NumberOfCreditInquiries=int(request.form.get("NumberOfCreditInquiries")),
            DebtToIncomeRatio=float(request.form.get("DebtToIncomeRatio")),
            BankruptcyHistory=int(request.form.get("BankruptcyHistory")),
            LoanPurpose=request.form.get("LoanPurpose"),
            PreviousLoanDefaults=int(request.form.get("PreviousLoanDefaults")),
            PaymentHistory=int(request.form.get("PaymentHistory")),
            LengthOfCreditHistory=int(request.form.get("LengthOfCreditHistory")),
            SavingsAccountBalance=int(request.form.get("SavingsAccountBalance")),
            CheckingAccountBalance=int(request.form.get("CheckingAccountBalance")),
            TotalAssets=int(request.form.get("TotalAssets")),
            TotalLiabilities=int(request.form.get("TotalLiabilities")),
            MonthlyIncome=float(request.form.get("MonthlyIncome")),
            UtilityBillsPaymentHistory=float(request.form.get("UtilityBillsPaymentHistory")),
            JobTenure=int(request.form.get("JobTenure")),
            NetWorth=int(request.form.get("NetWorth")),
            BaseInterestRate=float(request.form.get("BaseInterestRate")),
            InterestRate=float(request.form.get("InterestRate")),
            MonthlyLoanPayment=float(request.form.get("MonthlyLoanPayment")),
            TotalDebtToIncomeRatio=float(request.form.get("TotalDebtToIncomeRatio")),
        )
        final_data=data.get_data_as_data_frame()

        preprocessor= load_object("final_model/preprocessor.pkl")
        classification_model=load_object("final_model/model_classification.pkl")
        regression_model =load_object("final_model/model_regression.pkl")
        pre_data = preprocessor.transform(final_data)

        classification_pred = classification_model.predict(pre_data)


        risk_score_pred = regression_model.predict(pre_data)[0]  

        
        result_classification = "Rejected" if classification_pred[0] == 0 else "Approved"
        result_risk_score = f"{risk_score_pred:.2f}" 

        return render_template(
            "result.html",
            final_result=result_classification,
            risk_score=result_risk_score,
        )
if __name__ == '__main__':
    app.run(debug=True, port=8080)