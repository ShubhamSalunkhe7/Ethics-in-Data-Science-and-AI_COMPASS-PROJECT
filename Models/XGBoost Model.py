
# XGBoost Model (Gradient Boosting)
# _______________________________________________________

# SECTION 1 - Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 55)
print("  XGBOOST MODEL (GRADIENT BOOSTING)")
print("  COMPAS Fairness Audit")
print("=" * 55)



# SECTION 2 - Load Data
# _______________________________________________________

print("\n[STEP 1] Loading data...")

project_folder = Path(__file__).resolve().parent.parent
csv_file = project_folder / "Dataset" / "compas_cleaned.csv"

df = pd.read_csv(csv_file)

print(f"  Total rows: {len(df)}")
print(f"  Recidivism rate: {df['two_year_recid'].mean():.1%}")


# SECTION 3 - Features and Target
# _______________________________________________________

print("\n[STEP 2] Selecting features...")

# Same 4 features used consistently across ALL models
FEATURES  = ['age', 'priors_count', 'sex_male', 'charge_felony']
TARGET    = 'two_year_recid'
PROTECTED = 'race_binary'

X = df[FEATURES]
y = df[TARGET]
r = df[PROTECTED]

print(f"  Features: {FEATURES}")


# SECTION 4 - Train/Test Split
# _______________________________________________________

print("\n[STEP 3] Splitting data...")

# Identical split to all previous models
# RANDOM_SEED=42 ensures same 1,056 test cases
# for perfectly fair comparison

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, r,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"  Training rows: {len(X_train)}")
print(f"  Testing rows:  {len(X_test)}")

# NOTE: XGBoost also does NOT need feature scaling
# Like Random Forest, it uses trees internally
# Trees work on thresholds not distances

# SECTION 5 - Handle Class Imbalance
# _______________________________________________________

print("\n[STEP 4] Calculating class balance...")

# Count how many did NOT reoffend vs DID reoffend
count_not_reoffend = (y_train == 0).sum()
count_reoffend     = (y_train == 1).sum()

# scale_pos_weight tells XGBoost how much extra attention
# to pay to the minority class (reoffenders)
# Without this: model biased toward predicting 0 every time
scale_pos_weight = count_not_reoffend / count_reoffend

print(f"  Did NOT reoffend: {count_not_reoffend}")
print(f"  DID reoffend:     {count_reoffend}")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")


# SECTION 6 - Build and Train XGBoost
# _______________________________________________________

print("\n[STEP 5] Training XGBoost...")
print("  Building 300 gradient-boosted trees...")

model = XGBClassifier(

    # Number of trees (boosting rounds)
    n_estimators=300,

    # Max depth of each tree
    # Trees that are six levels deep allow us to find a lot of detail in our data
    # But if they're too many levels we run into the problem of memorising random junk.
    max_depth=6,

    # How much each tree corrects the previous one
    # If we make this value small (i.e., .05) then we get precision at the expense of speed (this is better than getting speed at the expense of accuracy).
    learning_rate=0.05,

    # Each tree will be trained on 80% of the rows from your training set randomly.
    # This adds some additional randomization to help keep you from over-fitting.
    subsample=0.8,

    # Randomly sample 80% of all available feature columns for use with each tree
    # More variety between trees
    colsample_bytree=0.8,

    # Compensates for class imbalance
    scale_pos_weight=scale_pos_weight,

    # Use logloss as the objective (standard for binary classification)
    eval_metric='logloss',

    # Set fixed seed for reproducability
    random_state=RANDOM_SEED,

    # Do not print out anything while training
    verbosity=0
)

model.fit(X_train, y_train)

print(f"  Training complete!")
print(f"  Trees built: {model.n_estimators}")
print(f"  Each tree max depth: {model.max_depth}")
print(f"  Learning rate: {model.learning_rate}")
