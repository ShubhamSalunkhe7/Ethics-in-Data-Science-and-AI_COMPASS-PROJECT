
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


# Split Data into Training and Testing
# _______________________________________________________

print("\n[STEP 3] Splitting data...")

# We split into:
# 80% TRAINING - the model learns from this
# 20% TESTING  - we test the model on data it has NEVER seen
#
# stratify= means the split keeps the same proportion of
# 0s and 1s in both halves (important for fair evaluation)

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, r,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"  Training rows: {len(X_train)}")
print(f"  Testing rows:  {len(X_test)}")
print(f"  Training recidivism rate: {y_train.mean():.1%}")
print(f"  Testing recidivism rate:  {y_test.mean():.1%}") 


# Scale the Features
# _______________________________________________________

print("\n[STEP 4] Scaling features...")

# StandardScaler transforms every number so that:
# Mean = 0, Standard Deviation = 1
#
# WHY? Logistic Regression works better when all numbers
# are on the same scale.
# Without this: age (25) and prior_crimes (3) have very
# different scales and confuse the model
# With this: both become small numbers like 0.3 or -1.2

scaler = StandardScaler()

# IMPORTANT: fit on TRAINING data only
# then transform BOTH training and testing
# Never fit on test data - that would be cheating
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("  Features scaled successfully")
print(f"  Example - Age mean before scaling: {X_train['age'].mean():.1f}")
print(f"  Example - Age mean after scaling:  {X_train_scaled[:,0].mean():.4f}")

