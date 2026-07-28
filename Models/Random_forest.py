
# Random Forest Model
# _______________________________________________________

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 55)
print("  RANDOM FOREST MODEL")
print("  COMPAS Fairness Audit")
print("=" * 55)



# Load the Cleaned Data
# _______________________________________________________

print("\n[STEP 1] Loading data...")

df = pd.read_csv("compas_cleaned.csv")

print(f"  Total rows: {len(df)}")
print(f"  Recidivism rate: {df['two_year_recid'].mean():.1%}")



# Choose Features and Target
# ______________________________________________________

print("\n[STEP 2] Selecting features...")

# Same 4 features as Logistic Regression baseline
# IMPORTANT: always use the same features across all models
# so comparisons are fair and consistent

FEATURES  = ['age', 'priors_count', 'sex_male', 'charge_felony']
TARGET    = 'two_year_recid'
PROTECTED = 'race_binary'

X = df[FEATURES]
y = df[TARGET]
r = df[PROTECTED]

print(f"  Features: {FEATURES}")
print(f"  X shape: {X.shape}")

