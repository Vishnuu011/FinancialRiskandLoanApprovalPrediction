from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Import your TrainingPipeline class
import sys
sys.path.append(r"C:\Users\HP\Desktop\FinancialRisk\FinancialRiskandLoanApprovalPrediction\src\loan_prediction\pipline\training_pipeline.py")
from src.loan_prediction.pipline.training_pipeline import TrainingPipeline 



# Instantiate the pipeline class
pipeline = TrainingPipeline()

def run_data_ingestion(**kwargs):
    return pipeline.start_data_ingestion()

def run_data_validation(ti, **kwargs):
    data_ingestion_artifact = ti.xcom_pull(task_ids='data_ingestion')
    return pipeline.start_data_validation(data_ingestion_artifact)

def run_data_transformation(ti, **kwargs):
    data_validation_artifact = ti.xcom_pull(task_ids='data_validation')
    return pipeline.start_data_transformation(data_validation_artifact)

def run_model_training(ti, **kwargs):
    data_transformation_artifact = ti.xcom_pull(task_ids='data_transformation')
    return pipeline.start_model_trainer(data_transformation_artifact)

with DAG(
    dag_id="training_pipeline_dag",
    description="Training pipeline orchestrated with Airflow",
    schedule_interval="@weekly",  
    start_date=datetime(2025, 1, 20)
) as dag:


    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_data_ingestion
    )

    data_validation = PythonOperator(
        task_id='data_validation',
        python_callable=run_data_validation,
        provide_context=True
    )

    data_transformation = PythonOperator(
        task_id='data_transformation',
        python_callable=run_data_transformation,
        provide_context=True
    )

    model_training = PythonOperator(
        task_id='model_training',
        python_callable=run_model_training,
        provide_context=True
    )


    data_ingestion >> data_validation >> data_transformation >> model_training
