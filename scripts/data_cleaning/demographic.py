# Import libraries
from pathlib import Path
import pandas as pd
from functions import *

# Load Data
BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "raw_data.xlsx"
DATA_PATH = BASE_DIR / "data" / "raw" / FILE_NAME

SHEET_NAME = "Demographic"
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)


# Categoreis Sanity Check
categorical_sanity_check(df, column='Gender',
                         valid_values=["Male", "Female"])



# Data Type Checking
expected_dtypes = {
    "Name":"str",
    "Gender" : "str",
    "Age" : "int64",
    "Salary" :"float64", 
    "LocationId" : "int64", 
    "Churned" : "int64"
}
validate_dtypes(df, expected_dtypes=expected_dtypes)


# Null Values Checker
missing_value_report(df)


# Outlier Detection and Distribution
plot_boxplot(df, "Salary")

plot_distribution(df, "Age")
plot_boxplot(df, "Age")


# Reomve 'Name' Column
df = df.drop(['Name'], axis='columns')


# Export file
OUTPUT_DIR = BASE_DIR / "data" / "processed"
file_name = "demographic.csv"
file_path = OUTPUT_DIR / file_name
df.to_csv(file_path ,index=False)