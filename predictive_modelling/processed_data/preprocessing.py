import numpy as np
import pandas as pd
import os
import pyodbc
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv


# Read From SQL Server Database
conn = pyodbc.connect(
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server={os.getenv('DB_SERVER')};"
    f"Database={os.getenv('DB_NAME')};"
    f"UID={os.getenv('DB_USER')};"
    f"PWD={os.getenv('DB_PASSWORD')};"
    "TrustServerCertificate=yes;"
)
cursor = conn.cursor()
cursor.fast_executemany = True


query = """
SELECT
    d.Gender, d.Age, d.Salary, l.[Geography] ,
    a.Tenure, a.Balance, a.NumProducts, a.HasCreditCard, a.IsActive,
    d.Churned
FROM demographic d
JOIN account a
    ON a.CustomerId = d.CustomerId
JOIN [location] l
    ON l.locationId = d.LocationId
"""

df = pd.read_sql(query, conn)

X = df.drop('Churned', axis=1)
y = df['Churned']

SEED = 200
T_SIZE = 0.2

df_train, df_test = train_test_split(
    df,
    test_size=T_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=df["Churned"]
)
# ===========================================
# Feature Engineering (Train)
# ===========================================
df_train["BalanceSalaryRatio"] = df_train.Balance / df_train.Salary
df_train["TenureByAge"] = df_train.Tenure / df_train.Age

# ===========================================
# Scaling Numerical Features (Train)
# ===========================================
num_cols = df_train.select_dtypes(include=['number']).columns.drop(['Churned'])

# Upgrade to RobustScaler for financial data
scaler = RobustScaler() 
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

# ===========================================
# Encoding Categorical Features (Train)
# ===========================================
cat_cols = df_train.select_dtypes(include=['string', 'object', 'bool']).columns.drop(['Churned'])
df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True, dtype=int)


# ===========================================
# Preprocessing Pipeline for Testing 
# (Self-contained function without global variables)
# ===========================================
def transform_test_data(df_test, fitted_scaler, numerical_columns, categorical_columns):
    # 1. Feature Engineering
    df_test['BalanceSalaryRatio'] = df_test.Balance / df_test.Salary
    df_test["TenureByAge"] = df_test.Tenure / df_test.Age

    # 2. Scaling (using the ALREADY FITTED scaler passed into the function)
    df_test[numerical_columns] = fitted_scaler.transform(df_test[numerical_columns])

    # 3. Encoding
    df_test = pd.get_dummies(df_test, columns=categorical_columns, drop_first=True, dtype=int)

    return df_test

# Call the function by explicitly passing the required objects
df_test = transform_test_data(
    df_test=df_test, 
    fitted_scaler=scaler, 
    numerical_columns=num_cols, 
    categorical_columns=cat_cols
)

# ===========================================
# Re-Order columns in Testing and Training 
# ===========================================

df_train.columns.equals(df_test.columns)
df_test = df_test[df_train.columns]

# ===========================================
# Declare Features and Target
# ===========================================

df_train_X = df_train.drop('Churned', axis='columns')
df_train_y = df_train['Churned']

df_test_X = df_test.drop('Churned', axis='columns')
df_test_y = df_test['Churned']

# ===========================================
# Save Data
# ===========================================

artifacts = {
    "X_train": df_train_X,
    "y_train": df_train_y,
    "X_test": df_test_X,
    "y_test": df_test_y
}

joblib.dump(artifacts, 'dataset_bundle.pkl')