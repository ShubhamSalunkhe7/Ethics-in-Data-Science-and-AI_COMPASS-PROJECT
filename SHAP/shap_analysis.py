# ________________________________________________________________

# SHAP Explainability Analysis

# ________________________________________________________________

# SECTION 1 — Imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # saves charts as files instead of
                        # opening pop-up windows
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("_" * 60)
print(" " * 10 + "SHAP EXPLAINABILITY ANALYSIS")
print("_" * 60)


# SECTION 2 — Load and Prepare Data
# ________________________________________________________________

print("\n(STEP 1) Loading data")

df = pd.read_csv("../Dataset/compas_cleaned.csv")

FEATURES = ['age', 'priors_count', 'sex_male', 'charge_felony']

# Human-readable names for charts
FEATURE_LABELS = {
    'age':           'Age',
    'priors_count':  'Prior Crimes',
    'sex_male':      'Sex (Male=1)',
    'charge_felony': 'Felony Charge (1=Yes)'
}

X = df[FEATURES]
y = df['two_year_recid']
r = df['race_binary']

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, r,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Rename columns for readable chart labels
X_test_display  = X_test.rename(columns=FEATURE_LABELS)
X_train_display = X_train.rename(columns=FEATURE_LABELS)

mask_black = (r_test == 1)
mask_white = (r_test == 0)

print(f"  Data loaded: {len(X_test)} test defendants")
print(f"  Black: {mask_black.sum()}  White: {mask_white.sum()}")


# SECTION 3 — Train the Models
# ________________________________________________________________

print("\n(STEP 2) Training models")

# XGBoost — primary model for SHAP analysis
# (best accuracy, native TreeExplainer support)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss', random_state=RANDOM_SEED, verbosity=0)
xgb.fit(X_train, y_train)
print("  ✓ XGBoost trained")

# Random Forest — for TreeExplainer comparison
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
print("  ✓ Random Forest trained")

# Logistic Regression — for LinearExplainer
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
lr.fit(X_train_scaled, y_train)
print("  ✓ Logistic Regression trained")


