
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


# Choose Features and Target
# _______________________________________________________

print("\n[STEP 2] Selecting features...")

# These are the 4 features the AI model will use to make predictions
# They match exactly what the dissertation uses
FEATURES = ['age', 'priors_count', 'sex_male', 'charge_felony']

# This is what we are trying to predict
# 1 = person reoffended within 2 years
# 0 = person did NOT reoffend
TARGET = 'two_year_recid'

# This is the protected attribute (used for fairness analysis)
PROTECTED = 'race_binary'

# Create our feature matrix X (the inputs)
X = df[FEATURES]

# Create our target vector y (the answers)
y = df[TARGET]

# Create our race column r (for fairness checks)
r = df[PROTECTED]

print(f"  Features used: {FEATURES}")
print(f"  Target column: {TARGET}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")



