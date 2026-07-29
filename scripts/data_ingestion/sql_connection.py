# Libraries and Packages
from pathlib import Path
import pandas as pd
import pyodbc
from dotenv import load_dotenv
import os

# Load environment variables FIRST
load_dotenv()

# ===========================================
# Load Processed Data Files
# ===========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR =  BASE_DIR / "data" / "processed"

# Load each dataset into its own specific DataFrame
df_demographic = pd.read_csv(DATA_DIR / "demographic.csv")
df_location = pd.read_csv(DATA_DIR / "location.csv")
df_account = pd.read_csv(DATA_DIR / "account.csv")

# ===========================================
# Create Database Connection
# ===========================================
conn = pyodbc.connect(
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server={os.getenv('DB_SERVER')};"
    f"Database={os.getenv('DB_NAME')};"
    f"UID={os.getenv('DB_USER')};"
    f"PWD={os.getenv('DB_PASSWORD')};"
    "TrustServerCertificate=yes;"
)
cursor = conn.cursor()

# ===========================================
# Push Data: Demographic Table
# ===========================================
print("Starting Demographic data insertion...")
cursor.execute("SET IDENTITY_INSERT demographic ON")
conn.commit()

query_demographic = """
INSERT INTO demographic (
    CustomerId, Gender, Age, Salary, LocationId, Churned
)
VALUES (?, ?, ?, ?, ?, ?)
"""

# Convert DataFrame to a list of tuples for bulk insert
data_demographic = [
    (
        int(row.CustomerId), 
        row.Gender, 
        int(row.Age), 
        float(row.Salary), 
        int(row.LocationId), 
        int(row.Churned)
    )
    for index, row in df_demographic.iterrows()
]

cursor.executemany(query_demographic, data_demographic)
conn.commit()
print("Demographic Data Inserted Successfully")

# ===========================================
# Push Data: Location Table
# ===========================================
print("Starting Location data insertion...")
cursor.execute("SET IDENTITY_INSERT location ON")
conn.commit()

query_location = """
INSERT INTO location (
    LocationId, Geography
)
VALUES (?, ?)
"""

data_location = [
    (
        int(row.LocationId), 
        row.Geography
    )
    for index, row in df_location.iterrows()
]

cursor.executemany(query_location, data_location)
conn.commit()
print("Location Data Inserted Successfully")

# ===========================================
# Push Data: Account Table
# ===========================================
print("Starting Account data insertion...")
cursor.execute("SET IDENTITY_INSERT account ON")
conn.commit()

query_account = """
INSERT INTO account (
    CustomerId, Tenure, Balance, NumProducts, HasCreditCard, IsActive
)
VALUES (?, ?, ?, ?, ?, ?)
"""

data_account = [
    (
        int(row.CustomerId),
        int(row.Tenure),
        float(row.Balance),
        int(row.NumProducts),
        int(row.HasCreditCard),
        int(row.IsActive)
    )
    for index, row in df_account.iterrows()
]

cursor.executemany(query_account, data_account)
conn.commit()
print("Account Data Inserted Successfully")

# Close the connection when finished
cursor.close()
conn.close()
print("Database connection closed.")