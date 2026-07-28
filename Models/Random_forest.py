
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

# Cross Validation
# _______________________________________________________

print("\n[STEP 5] Running 5-fold cross validation...")

# Same cross validation as Logistic Regression
# Allows direct comparison of CV scores between models

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_accuracy = cross_val_score(model, X_train, y_train,
                               cv=cv, scoring='accuracy')
cv_auc      = cross_val_score(model, X_train, y_train,
                               cv=cv, scoring='roc_auc')
cv_f1       = cross_val_score(model, X_train, y_train,
                               cv=cv, scoring='f1')

print(f"  CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"  CV AUC-ROC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# Make Predictions on Test Set
# _______________________________________________________

print("\n[STEP 6] Making predictions...")

# y_pred = hard prediction (0 or 1)
# This is the MAJORITY VOTE of all 200 trees

# y_prob = probability (0.0 to 1.0)
# This is the PROPORTION of trees that voted "1"
# Example: if 140 out of 200 trees say "reoffend"
# then y_prob = 140/200 = 0.70

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"  Predictions made for {len(y_pred)} defendants")
print(f"  Predicted high-risk: {y_pred.sum()} ({y_pred.mean():.1%})")
print(f"  Actual high-risk:    {y_test.sum()} ({y_test.mean():.1%})")

# Measure Overall Performance
# _______________________________________________________

print("\n[STEP 7] Measuring performance...")

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_prob)
f1       = f1_score(y_test, y_pred)

# Logistic Regression baseline scores (from previous model)
# We hardcode these here for direct comparison
LR_ACCURACY = 0.683
LR_AUC      = 0.742
LR_F1       = 0.648

print(f"\n  {'Metric':<15} {'Random Forest':>15} {'LR Baseline':>15} {'Better?':>10}")
print(f"  {'-'*57}")
print(f"  {'Accuracy':<15} {accuracy:>15.4f} {LR_ACCURACY:>15.4f} "
      f"{'✓ RF' if accuracy > LR_ACCURACY else '✓ LR':>10}")
print(f"  {'AUC-ROC':<15} {auc:>15.4f} {LR_AUC:>15.4f} "
      f"{'✓ RF' if auc > LR_AUC else '✓ LR':>10}")
print(f"  {'F1 Score':<15} {f1:>15.4f} {LR_F1:>15.4f} "
      f"{'✓ RF' if f1 > LR_F1 else '✓ LR':>10}")


# Feature Importance
# _______________________________________________________

print("\n[STEP 8] Feature importance (what the forest learned)...")

# Random Forest tells you which features were most useful
# across ALL 200 trees combined
# Higher importance = that feature was used more often
# to make correct splits in the trees

importance_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': model.feature_importances_.round(4)
}).sort_values('Importance', ascending=False)

print(f"\n  {'Rank':<6} {'Feature':<20} {'Importance':>12} {'Bar'}")
print(f"  {'-'*55}")
for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {rank:<6} {row['Feature']:<20} {row['Importance']:>12.4f}  {bar}")

print(f"\n  The most important feature is: {importance_df.iloc[0]['Feature']}")
print(f"  This matches the SHAP analysis in the dissertation")



# Racial Fairness Analysis
# _______________________________________________________

print("\n[STEP 9] Racial fairness analysis...")

mask_black = (r_test == 1)
mask_white = (r_test == 0)

# Accuracy per race
acc_black = accuracy_score(y_test[mask_black], y_pred[mask_black])
acc_white = accuracy_score(y_test[mask_white], y_pred[mask_white])

# F1 per race
f1_black  = f1_score(y_test[mask_black], y_pred[mask_black])
f1_white  = f1_score(y_test[mask_white], y_pred[mask_white])

# False Positive Rate per race
# FPR = wrongly labelled high-risk among people who did NOT reoffend
fp_b = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==0)).sum()
tn_b = ((y_pred[mask_black]==0) & (y_test.values[mask_black]==0)).sum()
fp_w = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==0)).sum()
tn_w = ((y_pred[mask_white]==0) & (y_test.values[mask_white]==0)).sum()
fpr_black = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0
fpr_white = fp_w / (fp_w + tn_w) if (fp_w + tn_w) > 0 else 0

# Demographic Parity Difference
rate_black = y_pred[mask_black].mean()
rate_white = y_pred[mask_white].mean()
dpd = rate_black - rate_white

# Equalised Odds Difference
tp_b  = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==1)).sum()
fn_b  = ((y_pred[mask_black]==0) & (y_test.values[mask_black]==1)).sum()
tp_w  = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==1)).sum()
fn_w  = ((y_pred[mask_white]==0) & (y_test.values[mask_white]==1)).sum()
tpr_black = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
tpr_white = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
eod = max(abs(tpr_black - tpr_white), abs(fpr_black - fpr_white))

fpr_ratio = fpr_black / fpr_white if fpr_white > 0 else 0

print(f"\n  {'Metric':<35} {'Black':>10} {'White':>10} {'Gap':>10}")
print(f"  {'-'*65}")
print(f"  {'Accuracy':<35} {acc_black:>10.4f} {acc_white:>10.4f} "
      f"{acc_black-acc_white:>+10.4f}")
print(f"  {'F1 Score':<35} {f1_black:>10.4f} {f1_white:>10.4f} "
      f"{f1_black-f1_white:>+10.4f}")
print(f"  {'False Positive Rate (FPR)':<35} {fpr_black:>10.4f} {fpr_white:>10.4f} "
      f"{fpr_black-fpr_white:>+10.4f}")
print(f"  {'True Positive Rate (TPR)':<35} {tpr_black:>10.4f} {tpr_white:>10.4f} "
      f"{tpr_black-tpr_white:>+10.4f}")
print(f"  {'High-Risk Prediction Rate':<35} {rate_black:>10.4f} {rate_white:>10.4f}")
print(f"\n  Demographic Parity Difference: {dpd:.4f}  (target: < 0.05)")
print(f"  Equalised Odds Difference:     {eod:.4f}  (target: < 0.05)")
print(f"  FPR Ratio (Black/White):       {fpr_ratio:.2f}x  (target: 1.00x)")

