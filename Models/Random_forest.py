
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

# Split Data into Training and Testing
# _______________________________________________________

print("\n[STEP 3] Splitting data (80% train, 20% test)...")

# Exact same split as Logistic Regression
# Using same RANDOM_SEED = 42 ensures IDENTICAL split
# This makes comparison between models perfectly fair

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, r,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"  Training rows: {len(X_train)}")
print(f"  Testing rows:  {len(X_test)}")

# NOTE: Random Forest does NOT need feature scaling
# Unlike Logistic Regression which needs StandardScaler,
# Random Forest uses decision trees which work on raw numbers
# Trees split data based on thresholds, not distances
# So age=25 and priors=3 work fine without scaling


# Build and Train the Random Forest
# _______________________________________________________

print("\n[STEP 4] Training Random Forest...")
print("  Building 200 decision trees...")

# n_estimators=200 means 200 individual decision trees
# Each tree votes. Majority wins.
# More trees = more stable but slower to train

# max_depth=10 means each tree can ask a maximum of
# 10 questions deep. This prevents overfitting.
# Without this limit trees grow until they memorise
# training data perfectly (bad for new data)

# min_samples_leaf=5 means each final leaf of the tree
# must have at least 5 training examples
# This stops the tree from making rules about just 1 person

# n_jobs=-1 means use ALL your CPU cores to train faster

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("  200 trees built and trained!")
print(f"  Total trees in forest: {model.n_estimators}")