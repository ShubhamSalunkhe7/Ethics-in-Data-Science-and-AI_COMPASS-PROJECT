#_______________________________________________________________________

# COMPAS FAIRNESS AUDIT
# Seven Fairness Metrics — All Four Models

#_______________________________________________________________________


# SECTION 1 — Import everything we need
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  SEVEN FAIRNESS METRICS — ALL FOUR MODELS")
print("  COMPAS Fairness Audit | COMP7039")
print("=" * 60)

# SECTION 2 — Load and prepare data
# (same as in all other model files) 
#_______________________________________________________________________

print("\n[STEP 1] Loading and preparing data...")

df = pd.read_csv("../Dataset/compas_cleaned.csv")

FEATURES  = ['age', 'priors_count', 'sex_male', 'charge_felony']
TARGET    = 'two_year_recid'
PROTECTED = 'race_binary'

X = df[FEATURES]
y = df[TARGET]
r = df[PROTECTED]

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, r,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# Scale for Logistic Regression and Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Masks for splitting results by race
mask_black = (r_test == 1)   # African-American
mask_white = (r_test == 0)   # Caucasian

print(f"  Test set: {len(X_test)} defendants")
print(f"  Black defendants in test: {mask_black.sum()}")
print(f"  White defendants in test: {mask_white.sum()}")

