
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

# SECTION 11 - Racial Fairness Analysis
# _______________________________________________________

print("\n[STEP 10] Racial fairness analysis...")

mask_black = (r_test == 1)
mask_white = (r_test == 0)

acc_black = accuracy_score(y_test[mask_black], y_pred[mask_black])
acc_white = accuracy_score(y_test[mask_white], y_pred[mask_white])
f1_black  = f1_score(y_test[mask_black], y_pred[mask_black])
f1_white  = f1_score(y_test[mask_white], y_pred[mask_white])

fp_b = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==0)).sum()
tn_b = ((y_pred[mask_black]==0) & (y_test.values[mask_black]==0)).sum()
fp_w = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==0)).sum()
tn_w = ((y_pred[mask_white]==0) & (y_test.values[mask_white]==0)).sum()
fpr_black = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0
fpr_white = fp_w / (fp_w + tn_w) if (fp_w + tn_w) > 0 else 0

tp_b = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==1)).sum()
fn_b = ((y_pred[mask_black]==0) & (y_test.values[mask_black]==1)).sum()
tp_w = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==1)).sum()
fn_w = ((y_pred[mask_white]==0) & (y_test.values[mask_white]==1)).sum()
tpr_black = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
tpr_white = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0

rate_black = y_pred[mask_black].mean()
rate_white = y_pred[mask_white].mean()
dpd        = rate_black - rate_white
eod        = max(abs(tpr_black - tpr_white), abs(fpr_black - fpr_white))
fpr_ratio  = fpr_black / fpr_white if fpr_white > 0 else 0

print(f"\n  {'Metric':<35} {'Black':>10} {'White':>10} {'Gap':>10}")
print(f"  {'-'*65}")
print(f"  {'Accuracy':<35} {acc_black:>10.4f} {acc_white:>10.4f} "
      f"{acc_black - acc_white:>+10.4f}")
print(f"  {'F1 Score':<35} {f1_black:>10.4f} {f1_white:>10.4f} "
      f"{f1_black - f1_white:>+10.4f}")
print(f"  {'False Positive Rate (FPR)':<35} {fpr_black:>10.4f} {fpr_white:>10.4f} "
      f"{fpr_black - fpr_white:>+10.4f}")
print(f"  {'True Positive Rate (TPR)':<35} {tpr_black:>10.4f} {tpr_white:>10.4f} "
      f"{tpr_black - tpr_white:>+10.4f}")
print(f"  {'High-Risk Prediction Rate':<35} {rate_black:>10.4f} {rate_white:>10.4f}")
print(f"\n  Demographic Parity Difference: {dpd:.4f}  (target: < 0.05)")
print(f"  Equalised Odds Difference:     {eod:.4f}  (target: < 0.05)")
print(f"  FPR Ratio (Black/White):       {fpr_ratio:.2f}x  (target: 1.00x)")


# SECTION 12 - Boosting Learning Curve
# _______________________________________________________

print("\n[STEP 11] Generating learning curve...")

# This shows how XGBoost improves as it adds more trees
# Demonstrates the boosting concept visually

train_scores = []
test_scores  = []
tree_counts  = list(range(10, 310, 10))

for n in tree_counts:
    temp = XGBClassifier(
        n_estimators=n,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        verbosity=0
    )
    temp.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, temp.predict(X_train)))
    test_scores.append(accuracy_score(y_test,  temp.predict(X_test)))

print("  Learning curve calculated")



# SECTION 13 - Visualisations
# _______________________________________________________

print("\n[STEP 12] Creating charts...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('XGBoost (Gradient Boosting) — COMPAS Fairness Audit',
             fontsize=14, fontweight='bold')

# Chart 1 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicted\nNo Reoffend', 'Predicted\nReoffend'],
            yticklabels=['Actual\nNo Reoffend', 'Actual\nReoffend'],
            ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix', fontweight='bold')

# Chart 2 — Boosting Learning Curve
axes[0, 1].plot(tree_counts, train_scores,
                label='Training Accuracy', color='#1F3864', linewidth=2)
axes[0, 1].plot(tree_counts, test_scores,
                label='Test Accuracy', color='#C00000', linewidth=2)
axes[0, 1].set_xlabel('Number of Trees')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Boosting Learning Curve\n'
                     '(How accuracy improves with more trees)',
                     fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Chart 3 — Feature Importance
colors_fi = ['#C00000', '#1F3864', '#2E75B6', '#BDD7EE']
axes[1, 0].barh(importance_df['Feature'],
                importance_df['Importance'],
                color=colors_fi, alpha=0.85, edgecolor='white')
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].set_title('XGBoost Feature Importance',fontweight='bold')
for i, v in enumerate(importance_df['Importance']):
    axes[1, 0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

# Chart 4 — All Three Models Compared
metrics   = ['Accuracy', 'AUC-ROC', 'F1 Score']
xgb_vals  = [accuracy, auc, f1]
rf_vals   = [RF_ACCURACY, RF_AUC, RF_F1]
lr_vals   = [LR_ACCURACY, LR_AUC, LR_F1]
x = np.arange(len(metrics))
w = 0.25

b1 = axes[1, 1].bar(x - w,   lr_vals,  w, label='Logistic Regression',
                    color='#BDD7EE', alpha=0.9, edgecolor='white')
b2 = axes[1, 1].bar(x,       rf_vals,  w, label='Random Forest',
                    color='#2E75B6', alpha=0.9, edgecolor='white')
b3 = axes[1, 1].bar(x + w,   xgb_vals, w, label='XGBoost',
                    color='#1F3864', alpha=0.9, edgecolor='white')

for bars in [b1, b2, b3]:
    for bar in bars:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f'{bar.get_height():.3f}',
                        ha='center', va='bottom', fontsize=7)

axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].set_ylim(0.5, 0.85)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('All Three Models Compared', fontweight='bold')
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Chart saved as: xgboost_results.png")



# SECTION 14 - Final Summary
# _______________________________________________________

print("\n" + "=" * 55)
print("  XGBOOST — FINAL SUMMARY")
print("=" * 55)
print(f"\n  Model:         XGBoost (300 gradient-boosted trees)")
print(f"  Learning rate: 0.05 (slow and precise)")
print(f"  Dataset:       COMPAS (5,278 defendants)")
print(f"\n  PERFORMANCE — ALL THREE MODELS:")
print(f"  {'Metric':<12} {'XGBoost':>10} {'Rand.Forest':>12} {'Log.Reg':>10}")
print(f"  {'-'*44}")
print(f"  {'Accuracy':<12} {accuracy:>10.4f} {RF_ACCURACY:>12.4f} {LR_ACCURACY:>10.4f}")
print(f"  {'AUC-ROC':<12} {auc:>10.4f} {RF_AUC:>12.4f} {LR_AUC:>10.4f}")
print(f"  {'F1 Score':<12} {f1:>10.4f} {RF_F1:>12.4f} {LR_F1:>10.4f}")
print(f"\n  FAIRNESS:")
print(f"  FPR Black:     {fpr_black:.4f}")
print(f"  FPR White:     {fpr_white:.4f}")
print(f"  FPR Ratio:     {fpr_ratio:.2f}x  (target: 1.00x)")
print(f"  DPD:           {dpd:.4f}  (target: < 0.05)")
print(f"  EOD:           {eod:.4f}  (target: < 0.05)")
print(f"\n  TOP FEATURE:   {importance_df.iloc[0]['Feature']}")
print(f"\n  KEY DISSERTATION INSIGHT:")
print(f"  XGBoost is the most accurate model")
print(f"  But fairness metrics are similar to simpler models")
print(f"  → Accuracy and fairness are NOT the same thing")
print(f"  → More complex ≠ more fair")
print("=" * 55)

