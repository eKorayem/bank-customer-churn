import numpy as np
import pandas as pd
import pyodbc
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split



# Read From SQL Server Database
conn = pyodbc.connect(
    "Driver={ODBC Driver 18 for SQL Server};" # Use 17 or 18 depending on your installed Linux driver
    "Server=localhost,1433;"
    "Database=BankChurn;"
    "UID=sa;"
    "PWD=SuperStrong!Project2026;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()


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
# Feature Engineering
# ===========================================

df_train["BalanceSalaryRatio"] = df_train.Balance / df_train.Salary
df_train["TenureByAge"] = df_train.Tenure / df_train.Age

# ===========================================
# Scaling Numerical Features
# ===========================================

num_cols = df_train.select_dtypes(include=['number']).columns
scaler = StandardScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

# ===========================================
# Encoding Categorical Features
# ===========================================

cat_cols = df_train.select_dtypes(include=['string', 'object', 'bool']).columns.drop(['Churned'])
df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True, dtype=int)

# ===========================================
# Preprocssing Pipeline in Testing
# ===========================================

def DfTestPipeline(df_test):
    df_test['BalanceSalaryRatio'] = df_test.Balance/df_test.Salary
    df_test["TenureByAge"] = df_test.Tenure / df_test.Age

    
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True, dtype=int)

    return df_test

df_test = DfTestPipeline(df_test=df_test)

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