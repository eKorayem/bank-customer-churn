from pathlib import Path
import pandas as pd
from functions import missing_value_report

# Set Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "raw_data.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "location.csv"

def process_location_data():
    # Load Data
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="Location")
    
    # Optional: Log missing values to standard output for pipeline monitoring
    print("Missing Values Report (Location):")
    print(missing_value_report(df))
    
    # Export file safely
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Location data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_location_data()