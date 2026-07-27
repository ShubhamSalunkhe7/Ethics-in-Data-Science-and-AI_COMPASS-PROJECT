
# COMPAS FAIRNESS AUDIT
# Logistic Regression Baseline Model
# _______________________________________________________

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score)
import warnings
warnings.filterwarnings("ignore")

# Fix random seed so results are same every time you run
# _______________________________________________________

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 55)
print("  LOGISTIC REGRESSION BASELINE")
print("  COMPAS Fairness Audit")
print("=" * 55)


# Load the Cleaned Data
# _______________________________________________________

print("\n[STEP 1] Loading data...")

from pathlib import Path

project_folder = Path(__file__).parent
csv_file = project_folder / "compas_cleaned.csv"

df = pd.read_csv(csv_file)

print(f"  Total rows loaded: {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Recidivism rate: {df['two_year_recid'].mean():.1%}")
