import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import __main__
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "documentation" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Map the scripts folder so Python can find preprocessing.py
sys.path.append(os.path.abspath(BASE_DIR / "scripts"))
from preprocessing import ChurnFeatureEngineer

# Manually inject the class into the main environment to bypass the joblib error
__main__.ChurnFeatureEngineer = ChurnFeatureEngineer

# Load data and model
data = joblib.load(BASE_DIR / "data" / "processed" / "dataset_bundle.pkl")
model = joblib.load(BASE_DIR / "data" / "processed" / "champion_model.pkl")
X_train = data['X_train']

# 1. Feature Importance Plot
feature_importances = model.feature_importances_
features = X_train.columns 

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features, hue=features, legend=False, palette="viridis")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png")
print("Saved feature_importance.png")