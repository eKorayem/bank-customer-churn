# Libraries and Pacages

from pathlib import Path
import pandas as pd
import pyodbc

BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "demographic.csv"
DATA_PATH =  BASE_DIR / "data" / "processed" / FILE_NAME
 
df = pd.read_csv(DATA_PATH)

# Create connetion

conn = pyodbc.connect(
    "Driver={ODBC Driver 18 for SQL Server};" # Use 17 or 18 depending on your installed Linux driver
    "Server=localhost,1433;"
    "Database=BankChurn;"
    "UID=sa;"
    "PWD=SuperStrong!Project2026;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()

# Push demographic to databse
cursor.execute("SET IDENTITY_INSERT demographic ON")
conn.commit()

query = """
INSERT INTO demographic (
    CustomerId,
    Gender,
    Age,
    Salary,
    LocationId,
    Churned
)
VALUES (?, ?, ?, ?, ?, ?)
"""

for index, row in df.iterrows():
    cursor.execute(
        query,
        int(row.CustomerId),
        row.Gender,
        int(row.Age),
        float(row.Salary),
        int(row.LocationId),
        int(row.Churned)
    )

conn.commit()
print("Data Inserted Successfully")



# Push Location to databse
cursor.execute("SET IDENTITY_INSERT location ON")
conn.commit()

query = """
INSERT INTO location (
    LocationId,
    Geography
)
VALUES (?, ?)
"""

for index, row in df.iterrows():
    cursor.execute(
        query,
        int(row.LocationId),
        row.Geography
    )

conn.commit()
print("Data Inserted Successfully")




# Push Account to databse
cursor.execute("SET IDENTITY_INSERT account ON")
conn.commit()

query = """
INSERT INTO account (
    CustomerId,
    Tenure,
    Balance,
    NumProducts,
    HasCreditCard,
    IsActive
)
VALUES (?, ?, ?, ?, ?, ?)
"""

for index, row in df.iterrows():
    cursor.execute(
        query,
        int(row.CustomerId),
        int(row.Tenure),
        float(row.Balance),
        int(row.NumProducts),
        int(row.HasCreditCard),
        int(row.IsActive)
    )

conn.commit()
print("Data Inserted Successfully")


df["Gender"].value_counts()