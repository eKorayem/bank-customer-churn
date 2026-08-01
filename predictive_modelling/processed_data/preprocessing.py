"""
Core Machine Learning Preprocessing Pipeline.
Handles data extraction, feature engineering, and scalable transformations.
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scripts.db_utils import get_db_connection # Reusing our centralized connection utility

# ===========================================
# Custom Feature Engineering Transformer
# ===========================================
class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer to generate custom financial ratios.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        # Create financial behavioral indicators
        X_new["BalanceSalaryRatio"] = X_new["Balance"] / X_new["Salary"].replace(0, np.nan)
        X_new["TenureByAge"] = X_new["Tenure"] / X_new["Age"].replace(0, np.nan)
        
        # Fill any potential infinite values created by division by zero
        X_new.fillna(0, inplace=True)
        return X_new

# ===========================================
# Data Extraction
# ===========================================
def fetch_raw_data() -> pd.DataFrame:
    """Extracts the joined analytical dataset from SQL Server."""
    query = """
    SELECT
        d.Gender, d.Age, d.Salary, l.[Geography],
        a.Tenure, a.Balance, a.NumProducts, a.HasCreditCard, a.IsActive,
        d.Churned
    FROM demographic d
    JOIN account a ON a.CustomerId = d.CustomerId
    JOIN [location] l ON l.locationId = d.LocationId
    """
    with get_db_connection() as conn:
        df = pd.read_sql(query, conn)
    return df

# ===========================================
# Pipeline Execution
# ===========================================
def main():
    print("Fetching raw data from database...")
    df = fetch_raw_data()

    # Define feature spaces
    X = df.drop('Churned', axis=1)
    y = df['Churned']

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=200, shuffle=True, stratify=y
    )

    # Identify column types automatically
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['string', 'object', 'bool']).columns.tolist()
    
    # We must add our engineered features to the numerical scaling list
    num_cols.extend(["BalanceSalaryRatio", "TenureByAge"])

    # Build the preprocessor column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, dtype=int), cat_cols)
        ],
        remainder='passthrough'
    )

    # Assemble the final execution pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', ChurnFeatureEngineer()),
        ('preprocessing', preprocessor)
    ])

    print("Fitting transformations and processing data...")
    # Fit strictly on training data to prevent data leakage, then transform
    X_train_processed = full_pipeline.fit_transform(X_train)
    
    # Transform test data using the parameters learned from training data
    X_test_processed = full_pipeline.transform(X_test)

    # Retrieve output feature names to maintain Pandas DataFrame structure
    feature_names = num_cols + full_pipeline.named_steps['preprocessing']\
                        .named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()

    df_train_X = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    df_test_X = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    # Package artifacts for modeling
    artifacts = {
        "X_train": df_train_X,
        "y_train": y_train,
        "X_test": df_test_X,
        "y_test": y_test,
        "pipeline": full_pipeline # Saving the pipeline itself is critical for future API deployment
    }

    output_path = 'dataset_bundle.pkl'
    joblib.dump(artifacts, output_path)
    print(f"✅ Pipeline executed successfully. Artifacts saved to {output_path}")

if __name__ == "__main__":
    main()