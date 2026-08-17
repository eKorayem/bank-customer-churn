# Bank Customer Churn Prediction & MLOps Pipeline

**[Live Demo: Bank Customer Churn Dashboard](https://bank-customer-churn-pu8ewpccplj2rpsmgebdly.streamlit.app/)**

An end-to-end machine learning and data engineering project designed to predict customer churn for financial institutions. This repository showcases production-grade practices, including secure SQL database ingestion, rigorous statistical validation, modular scikit-learn preprocessing pipelines, and automated model evaluation.

---

## Project Architecture & Workflow

```text
bank-churn-prediction/
├── data/                    # Local storage for raw and processed datasets (git-ignored)
├── notebooks/               # Exploratory Data Analysis and statistical testing notebooks
├── scripts/
│   ├── data_cleaning/       # Automated scripts for data type and outlier validation
│   ├── data_ingestion/      # High-performance bulk SQL insertion scripts
│   ├── db_utils.py          # Centralized database connection utility
│   ├── evaluation_script.py # Custom multi-metric model evaluation helper
│   ├── preprocessing.py     # Production Scikit-Learn pipeline and feature engineering
│   └── run_experiments.py   # Unified experiment runner and model comparison engine
├── requirements.txt         # Project dependencies
└── .gitignore               # Security exclusions

```

---

## Key Technical Highlights

* **Secure Data Engineering:** Ingests and processes relational banking data from a Microsoft SQL Server database using parameterized queries and environment variable security (`python-dotenv`).
* **High-Performance Ingestion:** Optimized database insertions by replacing standard row-by-row iteration with bulk tuple processing (`itertuples`) and `fast_executemany` to maximize execution efficiency.
* **Rigorous Statistical Testing:** Evaluated features using independent Welch's t-tests, Chi-square tests of independence, and one-way ANOVA, calculating effect sizes (Cohen's d, Cramér's V, Eta/Omega-squared) to quantify practical business impact.
* **Leakage-Free MLOps Pipeline:** Encapsulated custom feature engineering (`BalanceSalaryRatio`, `TenureByAge`, `IsHighRiskAge`) and robust scaling (`RobustScaler`) into a modular Scikit-Learn `Pipeline` and `ColumnTransformer` to prevent data leakage.
* **Unified Experimentation:** Automated hyperparameter tuning (`GridSearchCV`) across multiple classification algorithms (Logistic Regression, Random Forest, and XGBoost) with cross-validation and parallel CPU processing (`n_jobs=-1`).

---

## Exploratory Data Analysis & Statistical Insights

* **Balance Independence:** Welch's t-test ($p = 0.176$) and Cohen's $d = 0.036$ confirmed that raw account balance alone has no statistically significant or practical difference between churned and retained customers, highlighting that balance alone cannot explain churn behavior.
* **Activity Impact:** Chi-square contingency analysis ($p < 0.001$) and Cramér's V confirmed a statistically significant relationship between customer activity status (`IsActive`) and retention.
* **Tenure Consistency:** One-way ANOVA ($p = 0.470$) demonstrated that average customer tenure does not significantly vary across geographical regions.

---

## Model Performance Leaderboard

All models were evaluated on an isolated test set (20% holdout) using 5-fold cross-validation, optimizing for the **F1-Score** and **ROC-AUC** to balance precision and recall under class imbalance:

| Rank | Model | F1-Score | ROC-AUC | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| **1** | **Random Forest** | **0.584551** | 0.836238 | **0.687961** | **0.508167** |
| **2** | **XGBoost** | 0.579498 | **0.840124** | 0.680590 | 0.504554 |
| **3** | **Logistic Regression** | 0.463866 | 0.737176 | 0.678133 | 0.352490 |

> *Note: Support Vector Machines (Polynomial and RBF kernels) were benchmarked during exploratory phases and omitted from the final production search grid due to quadratic scaling overhead on CPU.*

---


## Interactive Prediction Dashboard

An interactive Python web application was built using Streamlit to translate the machine learning model's predictive insights into actionable intelligence for branch managers and retention teams. 

The application accepts live customer parameters, processes them through the automated feature engineering pipeline, and outputs a real-time churn probability calculation.

**To launch the dashboard locally:**
```bash
streamlit run app.py
```

---

## Getting Started & Installation

1. **Clone the repository:**

```bash
git clone [https://github.com/eKorayem/bank-customer-churn.git](https://github.com/eKorayem/bank-customer-churn.git)
cd bank-customer-churn

```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
Create a `.env` file in the root directory and configure your SQL Server credentials:

```env
DB_SERVER=localhost,1433
DB_NAME=BankChurn
DB_USER=your_username
DB_PASSWORD=your_password
```

4. **Run the pipeline:**
Execute the preprocessing and experiment runner scripts from your terminal:

```bash
python scripts/preprocessing.py
python scripts/run_experiments.py
```