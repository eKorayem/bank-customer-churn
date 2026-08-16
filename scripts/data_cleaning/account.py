# Import libraries
from pathlib import Path
import pandas as pd
from functions import *

# Load Data
BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "raw_data.xlsx"
DATA_PATH = BASE_DIR / "data" / "raw" / FILE_NAME

SHEET_NAME = "Account"
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

# Categories Sanity Check
categorical_sanity_check(df, column='IsActive',
                         valid_values=[0, 1])

# Data Type Checking
expected_dtypes = {
    "Tenure":"int64",
    "Balance" : "float64",
    "NumProducts" : "int64",
    "HasCreditCard" :"int64", 
    "IsActive" : "int64", 
}
validate_dtypes(df, expected_dtypes=expected_dtypes)

# Null Values Checker
missing_value_report(df)
df.columns

# Outlier Detection and Distribution
plot_boxplot(df, "Balance")


# Remove 'AccountId' Column
df = df.drop(['AccountId'], axis='columns')

# Export file
OUTPUT_DIR = BASE_DIR / "data" / "processed"
file_name = "account.csv"
file_path = OUTPUT_DIR / file_name
df.to_csv(file_path ,index=False)