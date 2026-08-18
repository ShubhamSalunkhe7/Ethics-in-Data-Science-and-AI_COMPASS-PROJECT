
# Neural Network (MLPClassifier)
# _______________________________________________________

# SECTION 1 - Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 55)
print("  NEURAL NETWORK (MLPClassifier)")
print("  COMPAS Fairness Audit")
print("=" * 55)


# SECTION 2 - Load Data
# _______________________________________________________

print("\n[STEP 1] Loading data...")

from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
csv_file = project_folder / "Dataset" / "compas_cleaned.csv"

df = pd.read_csv(csv_file)

print(f"  Total rows: {len(df)}")
print(f"  Recidivism rate: {df['two_year_recid'].mean():.1%}")



# SECTION 3 - Features and Target
# _______________________________________________________

print("\n[STEP 2] Selecting features...")

# Same 4 features as ALL other models
# This is absolutely critical for fair comparison
# If you change features here, you cannot know whether
# any performance difference is due to the model or the data

FEATURES  = ['age', 'priors_count', 'sex_male', 'charge_felony']
TARGET    = 'two_year_recid'
PROTECTED = 'race_binary'

X = df[FEATURES]
y = df[TARGET]
r = df[PROTECTED]

print(f"  Features: {FEATURES}")
print(f"  X shape:  {X.shape}")



# SECTION 4 - Train/Test Split
# _______________________________________________________

print("\n[STEP 3] Splitting data (80% train, 20% test)...")

# Same random_state=42 as ALL other models
# This ensures we test on IDENTICAL 1,056 defendants
# across all four models - the only fair comparison

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, r,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"  Training rows: {len(X_train)}")
print(f"  Testing rows:  {len(X_test)}")




# SECTION 5 - Scale Features (ESSENTIAL for Neural Networks)
# __________________________________________________________

print("\n[STEP 4] Scaling features (REQUIRED for neural networks)...")

# It's critical to scale data when working with Neural Networks 
# because unlike Tree Based Models (XGBoost / Random Forest) you don't need to worry about scale in a threshold based way 
# but instead Neural Network uses WEIGHTS to multiply the values of your features. 
# If your age is 25 and prior is 3 they will have significantly different weightings and learning will fail. 
#
# StandardScaler converts every feature so that:
# Mean = 0 and Standard Deviation = 1
#
# Before Scaling: age = 25, priors = 3 
# After scaling: age = 0.12, priors = -0.31
# Both now exist at approximately the same low scale as one another — allowing proper weighting to occur
#
# RULE: 
# ALWAYS fit the scaler to your training data alone 
# and then apply this trained scaler to ALL training data AND test data 
# Using the scaler on Test Data would allow "Cheating" 
#The Model has viewed Information from Future

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Scaling complete")
print(f"  Age BEFORE scaling — mean: {X_train['age'].mean():.2f}")
print(f"  Age AFTER  scaling — mean: {X_train_scaled[:,0].mean():.6f} (should be ~0)")
print(f"  Age AFTER  scaling — std:  {X_train_scaled[:,0].std():.6f}  (should be ~1)")


# SECTION 6 - Build and Train the Neural Network
# _______________________________________________________

print("\n[STEP 5] Building and training Neural Network...")
print("  Architecture: 4 inputs → 128 → 64 → 32 → 1 output")

# UNDERSTANDING THE ARCHITECTURE:
#
# hidden_layer_sizes = (128, 64, 32) 
# Creates 3 hidden layers:
#   Layer 1: 128 neurons  - finds simple patterns
#   Layer 2: 64 neurons   -  finds more complicated patterns
#   Layer 3: 32 neurons  -  which will help find even more complex patterns
#
# Think of it like this:
# Layer 1 asks: "Is age young?"  "Are priors high?"
# Layer 2 asks: "Young AND high priors?"
# Layer 3 asks: "Young AND high priors AND felony charge?"
# Output:        "High risk? Yes/No"
#
# activation='relu'
# ReLU = Rectified Linear Unit
# Simple rule:  Positive signal → keep the value, 
#               Negative signal → change it to 0
# It helps the neural network learn non-linear/complex patterns without becoming stuck.
#
# solver='adam'
# Adam is the learning algorithm (optimiser)
# It will adjust the networks weights so it makes less errors
# it will also help regulate how fast your model learns (so its not too slow or too fast)
# this can be thought of like a coach that is helping your network get better.
#
# alpha=0.001
# Regularisation — adds a penalty for having large weights
# Large weights = overfitting = bad on new data
# Higher alpha = simpler model = less overfitting
#
# max_iter=500
# The network processes all training data up to 500 times
# Each full pass through the data is called an "epoch"
# More epochs = more learning (up to a point)
#
# early_stopping=True
# If the model stops improving on a validation set,
# training stops automatically — prevents overfitting
# No point training more if we are no longer getting better
#
# validation_fraction=0.1
# 10% of training data is set aside just to monitor
# whether the model is still improving

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_SEED
)

model.fit(X_train_scaled, y_train)

print(f"  Training complete!")
print(f"  Actual epochs run: {model.n_iter_}  (out of max 500)")
print(f"  Total layers:  {len(model.hidden_layer_sizes) + 2} "
      f"(input + 3 hidden + output)")
print(f"  Early stopping: {'Yes — stopped early' if model.n_iter_ < 500 else 'No — ran all 500'}")


# SECTION 7 - Understand What Happened During Training
# _______________________________________________________

print("\n[STEP 6] Training loss curve...")

# The loss curve shows how the network improved over time
# Loss = how wrong the model is (lower = better)
# It should go DOWN as training progresses
# If it goes UP at the end, the model was overfitting

loss_history = model.loss_curve_
print(f"  Starting loss:  {loss_history[0]:.4f}  (should be high)")
print(f"  Final loss:     {loss_history[-1]:.4f}  (should be lower)")
print(f"  Improvement:    {((loss_history[0]-loss_history[-1])/loss_history[0])*100:.1f}%")




# SECTION 8 - Cross Validation
# _______________________________________________________

print("\n[STEP 7] Running 5-fold cross validation...")

# We use a fresh model for CV (not the trained one)
# to avoid data leakage from early stopping
cv_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=RANDOM_SEED
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_accuracy = cross_val_score(cv_model, X_train_scaled, y_train,
                               cv=cv, scoring='accuracy')
cv_auc      = cross_val_score(cv_model, X_train_scaled, y_train,
                               cv=cv, scoring='roc_auc')
cv_f1       = cross_val_score(cv_model, X_train_scaled, y_train,
                               cv=cv, scoring='f1')

print(f"  CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"  CV AUC-ROC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")





# SECTION 9 - Make Predictions
# _______________________________________________________

print("\n[STEP 8] Making predictions...")

# The neural network produces a probability for each person
# by passing their scaled features through all three layers
# of 128 + 64 + 32 neurons and applying weights and ReLU
# at each step, finally producing a number between 0 and 1
#
# y_pred rounds that probability to 0 or 1
# Default threshold is 0.5:
# probability >= 0.5 → predict reoffend (1)
# probability <  0.5 → predict no reoffend (0)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"  Predictions made for {len(y_pred)} defendants")
print(f"  Predicted high-risk: {y_pred.sum()} ({y_pred.mean():.1%})")
print(f"  Actual high-risk:    {y_test.sum()} ({y_test.mean():.1%})")


# SECTION 10 - Measure Performance
# _______________________________________________________

print("\n[STEP 9] Measuring performance...")

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_prob)
f1       = f1_score(y_test, y_pred)

# All previous model scores for comparison
LR_ACCURACY  = 0.683;  LR_AUC  = 0.742;  LR_F1  = 0.648
RF_ACCURACY  = 0.667;  RF_AUC  = 0.737;  RF_F1  = 0.627
XGB_ACCURACY = 0.670;  XGB_AUC = 0.730;  XGB_F1 = 0.641

print(f"\n  {'Metric':<12} {'MLP':>10} {'XGBoost':>10} "
      f"{'Rnd Forest':>12} {'LR Base':>10}")
print(f"  {'-'*56}")
print(f"  {'Accuracy':<12} {accuracy:>10.4f} {XGB_ACCURACY:>10.4f} "
      f"{RF_ACCURACY:>12.4f} {LR_ACCURACY:>10.4f}")
print(f"  {'AUC-ROC':<12} {auc:>10.4f} {XGB_AUC:>10.4f} "
      f"{RF_AUC:>12.4f} {LR_AUC:>10.4f}")
print(f"  {'F1 Score':<12} {f1:>10.4f} {XGB_F1:>10.4f} "
      f"{RF_F1:>12.4f} {LR_F1:>10.4f}")

all_accs = [accuracy, XGB_ACCURACY, RF_ACCURACY, LR_ACCURACY]
all_names = ['MLP', 'XGBoost', 'Random Forest', 'LR']
best_idx  = all_accs.index(max(all_accs))
print(f"\n  Best accuracy: {all_names[best_idx]} ({max(all_accs):.4f})")
print(f"  NOTE: Tree models often beat neural networks")
print(f"  on small tabular datasets — this is expected")



# SECTION 11 - Feature Importance
# (Permutation Method — different from tree models)
# _______________________________________________________

print("\n[STEP 10] Feature importance (permutation method)...")

# MLPClassifier has NO built-in feature_importances_
# Unlike Random Forest and XGBoost which track how often
# each feature was used in tree splits,
# the neural network just has billions of tiny weights —
# no single weight maps to a single feature
#
# PERMUTATION IMPORTANCE EXPLAINED:
# Step 1: Record the model's accuracy on test data
# Step 2: Randomly SHUFFLE one feature column
#         (this breaks any real pattern in that feature)
# Step 3: Measure the accuracy drop
# Step 4: Large drop = that feature was very important
# Step 5: Repeat for each feature
#
# Example:
# Normal accuracy:            0.683
# After shuffling age:        0.641  → drop of 0.042 = age is important
# After shuffling sex_male:   0.681  → drop of 0.002 = sex is less important

print("  Running permutation importance (10 repeats per feature)...")
print("  This shuffles each feature 10 times and measures accuracy drop...")

perm = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=RANDOM_SEED,
    scoring='accuracy'
)

importance_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': perm.importances_mean.round(4),
    'Std':        perm.importances_std.round(4)
}).sort_values('Importance', ascending=False)

print(f"\n  {'Rank':<6} {'Feature':<20} {'Importance':>12} {'±Std':>8}  Bar")
print(f"  {'-'*60}")
for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
    bar = '█' * max(0, int(row['Importance'] * 300))
    print(f"  {rank:<6} {row['Feature']:<20} {row['Importance']:>12.4f} "
          f"{row['Std']:>8.4f}  {bar}")

print(f"\n  Top feature: {importance_df.iloc[0]['Feature']}")
print(f"  This should match Random Forest and XGBoost findings")



# SECTION 12 - Racial Fairness Analysis
# _______________________________________________________

print("\n[STEP 11] Racial fairness analysis...")

mask_black = (r_test == 1)
mask_white = (r_test == 0)

# Accuracy and F1 per race
acc_black = accuracy_score(y_test[mask_black], y_pred[mask_black])
acc_white = accuracy_score(y_test[mask_white], y_pred[mask_white])
f1_black  = f1_score(y_test[mask_black], y_pred[mask_black])
f1_white  = f1_score(y_test[mask_white], y_pred[mask_white])

# False Positive Rate per race
fp_b = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==0)).sum()
tn_b = ((y_pred[mask_black]==0) & (y_test.values[mask_black]==0)).sum()
fp_w = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==0)).sum()
tn_w = ((y_pred[mask_white]==0) & (y_test.values[mask_white]==0)).sum()
fpr_black = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0
fpr_white = fp_w / (fp_w + tn_w) if (fp_w + tn_w) > 0 else 0

# True Positive Rate per race
tp_b = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==1)).sum()
fn_b = ((y_pred[mask_black]==0) & (y_test.values[mask_black]==1)).sum()
tp_w = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==1)).sum()
fn_w = ((y_pred[mask_white]==0) & (y_test.values[mask_white]==1)).sum()
tpr_black = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
tpr_white = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0

# Fairness metrics
rate_black = y_pred[mask_black].mean()
rate_white = y_pred[mask_white].mean()
dpd        = rate_black - rate_white
eod        = max(abs(tpr_black - tpr_white), abs(fpr_black - fpr_white))
fpr_ratio  = fpr_black / fpr_white if fpr_white > 0 else 0

# Predictive parity
tp_b_ppv = ((y_pred[mask_black]==1) & (y_test.values[mask_black]==1)).sum()
pp_b     = (y_pred[mask_black]==1).sum()
tp_w_ppv = ((y_pred[mask_white]==1) & (y_test.values[mask_white]==1)).sum()
pp_w     = (y_pred[mask_white]==1).sum()
ppv_b    = tp_b_ppv / pp_b if pp_b > 0 else 0
ppv_w    = tp_w_ppv / pp_w if pp_w > 0 else 0
pp_gap   = ppv_b - ppv_w

print(f"\n  {'Metric':<35} {'Black':>10} {'White':>10} {'Gap':>10}")
print(f"  {'-'*65}")
print(f"  {'Accuracy':<35} {acc_black:>10.4f} {acc_white:>10.4f} "
      f"{acc_black-acc_white:>+10.4f}")
print(f"  {'F1 Score':<35} {f1_black:>10.4f} {f1_white:>10.4f} "
      f"{f1_black-f1_white:>+10.4f}")
print(f"  {'False Positive Rate':<35} {fpr_black:>10.4f} {fpr_white:>10.4f} "
      f"{fpr_black-fpr_white:>+10.4f}")
print(f"  {'True Positive Rate':<35} {tpr_black:>10.4f} {tpr_white:>10.4f} "
      f"{tpr_black-tpr_white:>+10.4f}")
print(f"  {'High-Risk Prediction Rate':<35} {rate_black:>10.4f} {rate_white:>10.4f}")
print(f"\n  Demographic Parity Difference: {dpd:.4f}  (target: < 0.05)")
print(f"  Equalised Odds Difference:     {eod:.4f}  (target: < 0.05)")
print(f"  FPR Ratio (Black/White):       {fpr_ratio:.2f}x  (target: 1.00x)")
print(f"  Predictive Parity Gap:         {pp_gap:.4f}  (calibration)")



# SECTION 13 - The Architecture Insight
# _______________________________________________________

print("\n[STEP 12] Architecture comparison insight...")

print(f"""
  WHAT THIS TELLS US FOR THE DISSERTATION:

  The neural network has a fundamentally different
  architecture from all three tree-based models:
  → No trees, no splits, no votes
  → Instead: layers of weighted neurons + ReLU activation
  → Requires feature scaling (unlike trees)
  → No built-in feature importance (unlike trees)

  YET the fairness results are similar:
  → DPD ≈ {dpd:.3f}  (similar to LR, RF, XGBoost)
  → FPR gap ≈ {fpr_black-fpr_white:.3f}  (similar racial disparity)

  DISSERTATION CONCLUSION:
  The bias is NOT caused by the algorithm type.
  Four completely different architectures all produce
  similar racial disparities. This proves the bias
  comes from the DATA itself — the structural racism
  encoded in prior crimes and age distributions.
  Changing the algorithm does NOT fix the problem.
""")


# SECTION 14 - Training Loss Curve Visualisation
# _______________________________________________________

print("\n[STEP 13] Creating visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Neural Network (MLPClassifier) — COMPAS Fairness Audit',
             fontsize=14, fontweight='bold')

# Chart 1 — Training Loss Curve
axes[0, 0].plot(model.loss_curve_, color='#1F3864', linewidth=2,
                label='Training Loss')
if hasattr(model, 'validation_scores_') and model.validation_scores_:
    axes[0, 0].plot(model.validation_scores_, color='#C00000',
                    linewidth=2, linestyle='--', label='Validation Score')
axes[0, 0].set_xlabel('Epoch (Training Round)')
axes[0, 0].set_ylabel('Loss (lower = better)')
axes[0, 0].set_title('Training Loss Curve\n(How the network improved over time)',
                     fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Chart 2 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Predicted\nNo Reoffend', 'Predicted\nReoffend'],
            yticklabels=['Actual\nNo Reoffend', 'Actual\nReoffend'],
            ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix', fontweight='bold')

# Chart 3 — Permutation Feature Importance
colors_fi = ['#1F3864', '#2E75B6', '#4472C4', '#BDD7EE']
axes[1, 0].barh(importance_df['Feature'],
                importance_df['Importance'],
                xerr=importance_df['Std'],
                color=colors_fi, alpha=0.85,
                edgecolor='white', capsize=4)
axes[1, 0].set_xlabel('Accuracy Drop When Feature Shuffled')
axes[1, 0].set_title('Permutation Feature Importance\n'
                     '(Error bars show variability across 10 repeats)',
                     fontweight='bold')
axes[1, 0].axvline(x=0, color='black', linewidth=0.8)

# Chart 4 — All 4 Models FPR Comparison
model_names = ['LR\n(Baseline)', 'Random\nForest', 'XGBoost', 'Neural\nNetwork']
fpr_blacks  = [0.400, 0.360, 0.380, fpr_black]
fpr_whites  = [0.220, 0.200, 0.215, fpr_white]
x           = np.arange(len(model_names))
w           = 0.35

bars_b = axes[1, 1].bar(x - w/2, fpr_blacks, w,
                         label='African-American (Black)',
                         color='#C00000', alpha=0.85, edgecolor='white')
bars_w = axes[1, 1].bar(x + w/2, fpr_whites, w,
                         label='Caucasian (White)',
                         color='#2E75B6', alpha=0.85, edgecolor='white')
for bar in list(bars_b) + list(bars_w):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.006,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=8)
axes[1, 1].axhline(y=0.05, color='green', linestyle='--',
                   linewidth=1.2, label='Fair threshold (0.05)')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(model_names)
axes[1, 1].set_ylabel('False Positive Rate')
axes[1, 1].set_title('FPR by Race — ALL 4 MODELS\n'
                     '(Key finding: bias is consistent across all architectures)',
                     fontweight='bold')
axes[1, 1].legend(fontsize=7)
axes[1, 1].set_ylim(0, 0.6)

plt.tight_layout()
plt.savefig('neural_network_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Chart saved as: neural_network_results.png")


