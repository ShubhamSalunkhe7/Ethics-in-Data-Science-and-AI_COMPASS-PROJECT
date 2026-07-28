
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


# Train the Logistic Regression Model
# _______________________________________________________

print("\n[STEP 5] Training Logistic Regression model...")

# C=1.0 controls regularisation strength
# Higher C = model fits training data more tightly
# Lower C = simpler model, less risk of overfitting
# 1.0 is the default sensible starting point

# max_iter=1000 gives the model enough steps to converge
# (find the best weights)

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    random_state=RANDOM_SEED
)

# .fit() is where the actual learning happens
# The model looks at all training examples and adjusts
# its weights to minimise prediction errors
model.fit(X_train_scaled, y_train)

print("  Model trained successfully!")


# Cross Validation
# _______________________________________________________

print("\n[STEP 6] Running 5-fold cross validation...")

# Cross validation is a more reliable way to measure performance
# Instead of one 80/20 split, we do 5 different splits:
#
# Split 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
# Split 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
# Split 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
# Split 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
# Split 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]
#
# Then we average the 5 results for a more honest estimate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_accuracy = cross_val_score(model, X_train_scaled, y_train,
                               cv=cv, scoring='accuracy')
cv_auc      = cross_val_score(model, X_train_scaled, y_train,
                               cv=cv, scoring='roc_auc')
cv_f1       = cross_val_score(model, X_train_scaled, y_train,
                               cv=cv, scoring='f1')

print(f"  CV Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"  CV AUC-ROC:   {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  CV F1 Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")


# Make Predictions on Test Set
# _______________________________________________________

print("\n[STEP 7] Making predictions on test set...")

# y_pred = hard predictions (0 or 1)
# y_prob = probability of being class 1 (0.0 to 1.0)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"  Predictions made for {len(y_pred)} defendants")
print(f"  Predicted high-risk: {y_pred.sum()} ({y_pred.mean():.1%})")
print(f"  Actual high-risk:    {y_test.sum()} ({y_test.mean():.1%})")


# Measure Performance
# _______________________________________________________

print("\n[STEP 8] Measuring overall performance...")

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_prob)
f1       = f1_score(y_test, y_pred)

print(f"\n  OVERALL RESULTS:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"\n  (COMPAS benchmark accuracy ≈ 65%)")
print(f"  {'✓ BEATS COMPAS!' if accuracy > 0.65 else '✗ Below COMPAS'}")



# Racial Fairness Analysis
# _______________________________________________________

print("\n[STEP 9] Racial fairness analysis...")

# Split test results by race
mask_black = (r_test == 1)
mask_white = (r_test == 0)

# Overall accuracy per race
acc_black = accuracy_score(y_test[mask_black], y_pred[mask_black])
acc_white = accuracy_score(y_test[mask_white], y_pred[mask_white])

# F1 per race
f1_black  = f1_score(y_test[mask_black], y_pred[mask_black])
f1_white  = f1_score(y_test[mask_white], y_pred[mask_white])

# False Positive Rate per race
# FPR = out of people who did NOT reoffend, how many were
#        wrongly labelled as high-risk?
fp_black = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==0)).sum()
tn_black = ((y_pred[mask_black]==0) & (y_test.values[mask_black]==0)).sum()
fpr_black = fp_black / (fp_black + tn_black)

fp_white = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==0)).sum()
tn_white = ((y_pred[mask_white]==0) & (y_test.values[mask_white]==0)).sum()
fpr_white = fp_white / (fp_white + tn_white)

# Demographic Parity Difference
# Rate of high-risk prediction for Black minus White
rate_black = y_pred[mask_black].mean()
rate_white = y_pred[mask_white].mean()
dpd = rate_black - rate_white

print(f"\n  {'Metric':<35} {'Black':>10} {'White':>10}")
print(f"  {'-'*55}")
print(f"  {'Accuracy':<35} {acc_black:>10.4f} {acc_white:>10.4f}")
print(f"  {'F1 Score':<35} {f1_black:>10.4f} {f1_white:>10.4f}")
print(f"  {'False Positive Rate (FPR)':<35} {fpr_black:>10.4f} {fpr_white:>10.4f}")
print(f"  {'High-Risk Prediction Rate':<35} {rate_black:>10.4f} {rate_white:>10.4f}")
print(f"  {'Demographic Parity Diff':<35} {dpd:>10.4f} {'(target: < 0.05)':>10}")
print(f"\n  ProPublica reported FPR: Black ≈ 0.449, White ≈ 0.235")
print(f"  Our model FPR:           Black = {fpr_black:.3f}, White = {fpr_white:.3f}")
fpr_ratio = fpr_black / fpr_white if fpr_white > 0 else 0
print(f"  FPR Ratio (Black/White): {fpr_ratio:.2f}x")

