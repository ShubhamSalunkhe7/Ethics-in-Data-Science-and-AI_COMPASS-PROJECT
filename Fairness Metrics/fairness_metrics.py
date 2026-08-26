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

# SECTION 3 — Train all four models
#_______________________________________________________________________

print("\n[STEP 2] Training all four models...")

# Logistic Regression
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
lr.fit(X_train_scaled, y_train)
print("  ✓ Logistic Regression trained")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
print("  ✓ Random Forest trained")

# XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=6,  learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss', random_state=RANDOM_SEED, verbosity=0)
xgb.fit(X_train, y_train)
print("  ✓ XGBoost trained")

# Neural Network
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), activation='relu',
    solver='adam', alpha=0.001, max_iter=500,
    early_stopping=True, validation_fraction=0.1,
    random_state=RANDOM_SEED)
mlp.fit(X_train_scaled, y_train)
print("  ✓ Neural Network trained")

# Get predictions from all four models
predictions = {
    'Logistic Regression': {
        'y_pred': lr.predict(X_test_scaled),
        'y_prob': lr.predict_proba(X_test_scaled)[:, 1],
    },
    'Random Forest': {
        'y_pred': rf.predict(X_test),
        'y_prob': rf.predict_proba(X_test)[:, 1],
    },
    'XGBoost': {
        'y_pred': xgb.predict(X_test),
        'y_prob': xgb.predict_proba(X_test)[:, 1],
    },
    'Neural Network': {
        'y_pred': mlp.predict(X_test_scaled),
        'y_prob': mlp.predict_proba(X_test_scaled)[:, 1],
    },
}

print("\n  All predictions collected. Ready for fairness metrics.")

