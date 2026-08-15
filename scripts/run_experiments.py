"""
Unified Model Experimentation Pipeline.
Executes GridSearchCV across multiple algorithms and ranks them by performance.
"""
import warnings
import joblib
import pandas as pd
from pathlib import Path

# Scikit-Learn Models
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Custom Evaluation Module
from evaluation_script import evaluate_model
from preprocessing import ChurnFeatureEngineer

# ===========================================
# Secure File Pathing (Updated for new folder structure)
# ===========================================
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_bundle.pkl"

def load_data():
    """Loads the serialized preprocessed datasets."""
    print("Loading processed dataset...")
    data = joblib.load(DATA_PATH)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def run_all_experiments():
    X_train, X_test, y_train, y_test = load_data()

    # Calculate class imbalance ratio dynamically for XGBoost
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # ===========================================
    # Centralized Hyperparameter Grids
    # ===========================================
    experiments = {
        "Logistic Regression": {
            "model": LogisticRegression(solver='lbfgs', class_weight='balanced'),
            "params": {
                'C': [0.1, 0.5, 1, 10, 50, 100], 
                'max_iter': [250], 
                'penalty': ['l2']
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight="balanced"),
            "params": {
                'max_depth': [3, 5, 6, 7, 8], 
                'n_estimators': [50, 100],
                'min_samples_split': [3, 5, 6, 7]
            }
        },
        # "SVM (Polynomial)": {
        #     "model": SVC(class_weight="balanced", probability=True, kernel='poly'),
        #     "params": {
        #         'C': [0.5, 1, 10, 50, 100], 
        #         'gamma': [0.1, 0.01, 0.001],
        #         'degree': [2, 3]
        #     }
        # },
        # "SVM (RBF)": {
        #     "model": SVC(class_weight="balanced", probability=True, kernel='rbf'),
        #     "params": {
        #         'C': [0.5, 100, 150], 
        #         'gamma': [0.1, 0.01, 0.001]
        #     }
        # },
        "XGBoost": {
            "model": XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42),
            "params": {
                'max_depth': [5, 6, 7, 8], 
                'learning_rate': [0.05, 0.1, 0.2, 0.3], 
                'n_estimators': [5, 10, 20, 100],
                'scale_pos_weight': [1, imbalance_ratio]
            }
        }
    }

    results = []

    # ===========================================
    # Unified Training Loop
    # ===========================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for name, config in experiments.items():
            print(f"Starting Training for: {name}...")
            
            grid = GridSearchCV(
                estimator=config["model"], 
                param_grid=config["params"], 
                cv=5,         
                refit=True, 
                scoring='f1',
                verbose=0,
                n_jobs=-1     
            )
            
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            
            metrics = evaluate_model(best_model, X_test, y_test)
            metrics["Model"] = name
            
            results.append(metrics)
            print(f"Finished {name} | F1 Score: {metrics['F1_Churn']:.4f}\n")

    # ===========================================
    # Performance Leaderboard
    # ===========================================
    results_df = pd.DataFrame(results).set_index("Model")
    
    print("-" * 50)
    print("FINAL MODEL LEADERBOARD (Ranked by F1 Score)")

    # Save the Random Forest model for the Streamlit App
    rf_best_model = experiments["Random Forest"]["model"]
    # We have to re-fit it quickly on the full training data just to be safe
    rf_best_model.fit(X_train, y_train)
    joblib.dump(rf_best_model, 'data/processed/champion_model.pkl')
    print("Champion Random Forest model saved to data/processed/champion_model.pkl")
    print("-" * 50)
    
    leaderboard = results_df[["F1_Churn", "ROC_AUC", "Recall_Churn", "Precision_Churn"]].sort_values(by="F1_Churn", ascending=False)
    print(leaderboard)

if __name__ == "__main__":
    run_all_experiments()