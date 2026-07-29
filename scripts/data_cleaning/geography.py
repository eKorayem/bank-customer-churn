# Import libraries
from pathlib import Path
import pandas as pd
from functions import *

# Load Data
BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "raw_data.xlsx"
DATA_PATH = BASE_DIR / "data" / "raw" / FILE_NAME

SHEET_NAME = "Location"
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

pth_acc = BASE_DIR / "data" / "processed" / "account.csv"
pth_demo = BASE_DIR / "data" / "processed" / "demographic.csv"


# Export file
OUTPUT_DIR = BASE_DIR / "data" / "processed"
file_name = "location.csv"
file_path = OUTPUT_DIR / file_name
df.to_csv(file_path ,index=False)