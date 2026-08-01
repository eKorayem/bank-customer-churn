import os
import pyodbc
from dotenv import load_dotenv

# Load environment variables once upon module import
load_dotenv()

def get_db_connection() -> pyodbc.Connection:
    """
    Establishes and returns a secure connection to the SQL Server database.
    """
    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server={os.getenv('DB_SERVER')};"
        f"Database={os.getenv('DB_NAME')};"
        f"UID={os.getenv('DB_USER')};"
        f"PWD={os.getenv('DB_PASSWORD')};"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)