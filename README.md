# 💰 Financial Risk and Loan Approval Prediction

An **end-to-end Machine Learning and MLOps project** designed to predict **loan approval** and **financial risk levels** for applicants based on demographic, financial, and historical data.  
This project includes **data pipelines**, **Airflow DAGs**, **MLflow tracking**, **Docker deployment**, and **Flask web integration**, demonstrating full lifecycle management of a production-grade ML model.

---

## 🚀 Project Overview

This project predicts whether a loan should be approved and estimates the associated financial risk.  
It is designed using a **modular architecture** — from data ingestion to model deployment — and integrates **MLflow**, **Apache Airflow**, and **Docker** for complete MLOps automation.

**Key Highlights**

- 🧩 Modular ML pipeline (Data Ingestion → Validation → Transformation → Training → Prediction)
- 🧮 Supports both **Classification** and **Regression** models
- ⚙️ Automated pipeline scheduling with **Apache Airflow**
- 📊 Experiment tracking and model versioning using **MLflow**
- ☁️ Real-time predictions via **Flask Web App**
- 🐳 Fully containerized with **Docker**
- 📂 MongoDB integration for dynamic data and logging

---

## 🧠 Key Features

✅ **Complete ML Pipeline**

- Data ingestion, validation, transformation, model training, and evaluation.
- Automatic artifact tracking and reproducibility.

✅ **MLflow + DagsHub Integration**

- Model metrics and artifacts are tracked with MLflow.
- All experiments logged on [DagsHub](https://dagshub.com/Vishnuu011/FinancialRiskandLoanApprovalPrediction.mlflow).

✅ **Dual Model System**

- Classification → Loan approval prediction
- Regression → Risk score estimation

✅ **Automatic Model Selection**

- Evaluates multiple algorithms and selects the best-performing one.

✅ **Custom Exception & Logging System**

- Handles runtime errors gracefully and logs important information.

✅ **Reusable Utilities**

- Custom modules for saving/loading objects, evaluating models, and tracking metrics.

---

## 🏗️ Project Structure

```
├── 📁 .github
│   └── 📁 workflows
│       ├── ⚙️ jekyll-gh-pages.yml
│       ├── ⚙️ main.yaml
│       └── ⚙️ static.yml
├── 📁 FinancialRiskData
│   └── 📄 Loan.csv
├── 📁 config
│   └── ⚙️ schema.yaml
├── 📁 dags
│   ├── 🐍 __init__.py
│   └── 🐍 ml_pipeline_dag.py
├── 📁 final_model
│   ├── 📄 model_classification.pkl
│   ├── 📄 model_regression.pkl
│   └── 📄 preprocessor.pkl
├── 📁 src
│   ├── 📁 loan_prediction
│   │   ├── 📁 cloud
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 components
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 data_ingestion.py
│   │   │   ├── 🐍 data_tansformation.py
│   │   │   ├── 🐍 data_validation.py
│   │   │   └── 🐍 model_trainer.py
│   │   ├── 📁 constant
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 entity
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 artifact_entity.py
│   │   │   └── 🐍 config_entity.py
│   │   ├── 📁 exception
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 logger
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 pipline
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 prediction_pipeline.py
│   │   │   └── 🐍 training_pipeline.py
│   │   ├── 📁 utils
│   │   │   ├── 📁 ml_utils
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 estimator.py
│   │   │   │   └── 🐍 metrics.py
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 utils.py
│   │   └── 🐍 __init__.py
│   └── 🐍 __init__.py
├── 📁 static
│   ├── 🎨 result.css
│   └── 🎨 styles.css
├── 📁 templates
│   ├── 🌐 index.html
│   └── 🌐 result.html
├── ⚙️ .dockerignore
├── ⚙️ .gitignore
├── 🐳 Dockerfile
├── 📄 LICENSE
├── 📝 README.md
├── 🐍 app.py
├── 🐍 demo.py
├── ⚙️ docker-compose.yml
├── 📄 loan_predictions.csv
├── 🐍 main.py
├── 🐍 push_data.py
├── 📄 requirements.txt
├── 🐍 setup.py
├── 🐍 templates.py
├── 🐍 testmongodb.py
└── 🐍 user_input_mongo_db.py
```

## ⚙️ Tech Stack

| Component               | Technology                                    |
| ----------------------- | --------------------------------------------- |
| **Language**            | Python 3.9+                                   |
| **Libraries**           | NumPy, Pandas, Scikit-learn, MLflow, DagsHub  |
| **Experiment Tracking** | MLflow + DagsHub                              |
| **Visualization**       | Matplotlib / Seaborn                          |
| **Model Storage**       | Pickle / MLflow Registry                      |
| **Logging**             | Custom Python Logger                          |
| **Pipeline Design**     | Modular architecture with reusable components |

---

# 🧰 How to Run the Project

# 🪜 1. Clone the Repository

```bash
git clone https://github.com/Vishnuu011/FinancialRiskandLoanApprovalPrediction.git
cd FinancialRiskandLoanApprovalPrediction
```

# ⚙️ 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\\Scripts\\activate   # On Windows
# source venv/bin/activate  # On Mac/Linux
```

# 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

# 🧠 4. Run the Training Pipeline

```bash
python src/loan_prediction/pipeline/training_pipeline.py
```

# 🧾 Outputs

# #📂 Artifacts Generated

- transformed_train.npy, transformed_test.npy

- model_classification.pkl

- model_regression.pkl

- MLflow Experiment Runs

- logs/training.log

# 🧑‍💻 Author

# 👨‍💻 Vishnu

Machine Learning Engineer | AI Research Enthusiast
📍 Building intelligent, explainable, and production-ready ML systems
