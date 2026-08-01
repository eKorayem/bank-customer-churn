"""
Ingests processed CSV data into the SQL Server database securely and efficiently.
"""
from pathlib import Path
import pandas as pd
from scripts.db_utils import get_db_connection # Using our new centralized utility

# ===========================================
# Setup & Load Data
# ===========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

# Load datasets
datasets = {
    "demographic": pd.read_csv(DATA_DIR / "demographic.csv"),
    "location": pd.read_csv(DATA_DIR / "location.csv"),
    "account": pd.read_csv(DATA_DIR / "account.csv")
}

# ===========================================
# Database Ingestion Operations
# ===========================================
def bulk_insert_table(cursor, table_name: str, df: pd.DataFrame, query: str):
    """
    Handles identity insertion and bulk executes data into a specified table.
    """
    print(f"Starting {table_name} data insertion...")
    cursor.execute(f"SET IDENTITY_INSERT {table_name} ON")
    
    # itertuples() is exponentially faster than iterrows() for tuple generation
    # Ensure dataframe columns match the exact query insertion order before this step
    data_tuples = list(df.itertuples(index=False, name=None))
    
    cursor.executemany(query, data_tuples)
    print(f"{table_name.capitalize()} Data Inserted Successfully")

def main():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.fast_executemany = True # Excellent inclusion for ODBC optimization

    try:
        # 1. Demographic Insertion
        q_demo = """INSERT INTO demographic (CustomerId, Gender, Age, Salary, LocationId, Churned) VALUES (?, ?, ?, ?, ?, ?)"""
        # Ensure exact column order matches query
        df_demo = datasets["demographic"][['CustomerId', 'Gender', 'Age', 'Salary', 'LocationId', 'Churned']]
        bulk_insert_table(cursor, "demographic", df_demo, q_demo)

        # 2. Location Insertion
        q_loc = """INSERT INTO location (LocationId, Geography) VALUES (?, ?)"""
        df_loc = datasets["location"][['LocationId', 'Geography']]
        bulk_insert_table(cursor, "location", df_loc, q_loc)

        # 3. Account Insertion
        q_acc = """INSERT INTO account (CustomerId, Tenure, Balance, NumProducts, HasCreditCard, IsActive) VALUES (?, ?, ?, ?, ?, ?)"""
        df_acc = datasets["account"][['CustomerId', 'Tenure', 'Balance', 'NumProducts', 'HasCreditCard', 'IsActive']]
        bulk_insert_table(cursor, "account", df_acc, q_acc)

        # Commit all successful transactions
        conn.commit()
        
    except Exception as e:
        conn.rollback() # Rollback on failure to prevent partial commits
        print(f"Database insertion failed: {e}")
    finally:
        cursor.close()
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()