
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



# SECTION 7 - Cross Validation
# _______________________________________________________

print("\n[STEP 6] Running 5-fold cross validation...")

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




# SECTION 8 - Make Predictions
# _______________________________________________________

print("\n[STEP 7] Making predictions...")

# y_pred = final 0 or 1 decision
# This is the SUM of all 300 trees combined

# y_prob = probability score between 0.0 and 1.0
# Higher score = model more confident this person will reoffend

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"  Predictions made for {len(y_pred)} defendants")
print(f"  Predicted high-risk: {y_pred.sum()} ({y_pred.mean():.1%})")
print(f"  Actual high-risk:    {y_test.sum()} ({y_test.mean():.1%})")



# SECTION 9 - Overall Performance
# _______________________________________________________

print("\n[STEP 8] Measuring performance...")

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_prob)
f1       = f1_score(y_test, y_pred)

# Previous model scores for comparison
LR_ACCURACY = 0.683
LR_AUC      = 0.742
LR_F1       = 0.648
RF_ACCURACY = 0.667
RF_AUC      = 0.737
RF_F1       = 0.627

print(f"\n  {'Metric':<12} {'XGBoost':>12} {'Rand.Forest':>12} {'Log.Reg':>12}")
print(f"  {'-'*48}")
print(f"  {'Accuracy':<12} {accuracy:>12.4f} {RF_ACCURACY:>12.4f} {LR_ACCURACY:>12.4f}")
print(f"  {'AUC-ROC':<12} {auc:>12.4f} {RF_AUC:>12.4f} {LR_AUC:>12.4f}")
print(f"  {'F1 Score':<12} {f1:>12.4f} {RF_F1:>12.4f} {LR_F1:>12.4f}")

best_acc = max(accuracy, RF_ACCURACY, LR_ACCURACY)
winner   = 'XGBoost' if accuracy == best_acc else ('RF' if RF_ACCURACY == best_acc else 'LR')
print(f"\n  Best accuracy: {winner} ({best_acc:.4f})")



# SECTION 10 - Feature Importance
# _______________________________________________________

print("\n[STEP 9] Feature importance...")

# XGBoost calculates feature importance differently to Random Forest
# It measures: how much did each feature REDUCE errors
# across all 300 trees combined?

importance_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': model.feature_importances_.round(4)
}).sort_values('Importance', ascending=False)

print(f"\n  {'Rank':<6} {'Feature':<20} {'Importance':>12} {'Bar'}")
print(f"  {'-'*55}")
for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {rank:<6} {row['Feature']:<20} {row['Importance']:>12.4f}  {bar}")

print(f"\n  Most important: {importance_df.iloc[0]['Feature']}")
print(f"  This is the feature XGBoost used most often")
print(f"  to reduce prediction errors across 300 trees")

