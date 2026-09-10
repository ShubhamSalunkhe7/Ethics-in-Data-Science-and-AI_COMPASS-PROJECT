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


# SECTION 4 — XGBoost SHAP (TreeExplainer)
# GLOBAL ANALYSIS — all 1,056 test defendants
# ________________________________________________________________

print("\n[STEP 3] Running XGBoost SHAP (TreeExplainer)...")
print("  Computing SHAP values for all test defendants...")

# TreeExplainer is the fast, exact explainer for tree models
# It can compute SHAP values for all 1,056 defendants
# in just a few seconds
explainer_xgb = shap.TreeExplainer(xgb)

# shap_values contains one SHAP value per person per feature
# Shape: (1056, 4) — 1056 defendants, 4 features
shap_values_xgb = explainer_xgb.shap_values(X_test)

print(f"  SHAP values computed: shape = {shap_values_xgb.shape}")
print(f"  (rows=defendants, columns=features)")

# Mean absolute SHAP per feature (global importance)
mean_abs_shap = np.abs(shap_values_xgb).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature':    [FEATURE_LABELS[f] for f in FEATURES],
    'Mean_SHAP':  mean_abs_shap.round(4)
}).sort_values('Mean_SHAP', ascending=False)

print(f"\n  GLOBAL FEATURE IMPORTANCE (Mean |SHAP|):")
print(f"  {'Rank':<6} {'Feature':<25} {'Mean |SHAP|':>12}  Bar")
print(f"  {'-'*60}")
for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
    bar = '█' * int(row['Mean_SHAP'] * 80)
    print(f"  {rank:<6} {row['Feature']:<25} "
          f"{row['Mean_SHAP']:>12.4f}  {bar}")


# SECTION 5 — CHART 1: Beeswarm Plot
# One dot per defendant, shows direction AND magnitude
# ________________________________________________________________

print("\n(STEP 4) Creating Chart 1 — Beeswarm plot")

plt.figure(figsize=(10, 5))

# summary_plot with plot_type='dot' creates the beeswarm
# Each dot = one defendant
# Position on x-axis = SHAP value (left=low risk, right=high risk)
# Colour = actual feature value (blue=low, red=high)
shap.summary_plot(
    shap_values_xgb,
    X_test_display,
    plot_type='dot',
    show=False,
    max_display=10
)

plt.title(
    'SHAP Beeswarm Plot — XGBoost\n'"
    'Each dot = one defendant  |  '
    'Red = high feature value  |  '
    'Position = impact on prediction',
    fontsize=10, pad=12
)
plt.xlabel('SHAP Value (negative = lower risk, positive = higher risk)')
plt.tight_layout()
plt.savefig('shap_beeswarm_xgb.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: shap_beeswarm_xgb.png")


# SECTION 6 — CHART 2: Bar Plot (Mean Absolute SHAP)
# Simple overall feature importance ranking
# ________________________________________________________________

print("\n(STEP 5) Creating Chart 2 — Bar plot...")

plt.figure(figsize=(8, 4))

shap.summary_plot(
    shap_values_xgb,
    X_test_display,
    plot_type='bar',
    show=False,
    max_display=10
)

plt.title(
    'Mean |SHAP| Feature Importance — XGBoost\n'
    'Higher bar = feature has larger average impact on predictions',
    fontsize=10, pad=10
)
plt.tight_layout()
plt.savefig('shap_bar_xgb.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: shap_bar_xgb.png")


# SECTION 7 — CHART 3: Waterfall Plots (Local Analysis)
# Three specific defendants — shows individual explanations
# ________________________________________________________________

print("\n(STEP 6) Creating Chart 3 — Waterfall plots (3 individuals)")

# Get predicted probabilities
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = xgb.predict(X_test)
r_arr      = r_test.values

# We need the SHAP Explanation object (not just values)
# for waterfall plots
shap_explanation = explainer_xgb(X_test_display)

# ── Find three representative defendants ─────────────────────

# Case A — Black defendant predicted HIGH RISK
# (most common victim of the bias we are studying)
case_a_candidates = np.where(
    (r_arr == 1) &           # Black defendant
    (y_pred_xgb == 1) &      # Predicted high risk
    (y_prob_xgb > 0.65)      # High confidence
)[0]
idx_a = case_a_candidates[0] if len(case_a_candidates) > 0 else 0

# Case B — White defendant predicted HIGH RISK
# (compare with Case A — same prediction, different race)
case_b_candidates = np.where(
    (r_arr == 0) &           # White defendant
    (y_pred_xgb == 1) &      # Predicted high risk
    (y_prob_xgb > 0.65)      # High confidence
)[0]
idx_b = case_b_candidates[0] if len(case_b_candidates) > 0 else 1

# Case C — Borderline defendant (probability closest to 0.5)
# (shows the uncertainty zone where small changes matter)
idx_c = int(np.argmin(np.abs(y_prob_xgb - 0.5)))

cases = [
    (idx_a, f"Case A — African-American, Predicted HIGH RISK"),
    (idx_b, f"Case B — Caucasian, Predicted HIGH RISK"),
    (idx_c, f"Case C — Borderline Prediction (prob ≈ 0.50)"),
]

for idx, title in cases:
    try:
        plt.figure(figsize=(9, 4))
        shap.waterfall_plot(
            shap_explanation[idx],
            show=False,
            max_display=10
        )
        race_label = 'Black' if r_arr[idx] == 1 else 'White'
        plt.title(
            f'SHAP Waterfall — {title}\n'
            f'Predicted probability: {y_prob_xgb[idx]:.3f}  |  '
            f'Actual outcome: {y_test.values[idx]}  |  '
            f'Race: {race_label}',
            fontsize=9, pad=8
        )
        plt.tight_layout()
        fname = f"shap_waterfall_case_{title[5]}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {fname}")
        print(f"    Race={race_label}  "
              f"Prob={y_prob_xgb[idx]:.3f}  "
              f"Actual={y_test.values[idx]}")

        # Print the SHAP values for this person in plain text
        print(f"    Feature contributions for this defendant:")
        sv = shap_values_xgb[idx]
        for feat, val in zip(FEATURES, sv):
            direction = '→ HIGH RISK' if val > 0 else '→ LOW RISK'
            bar = '█' * int(abs(val) * 40)
            print(f"      {FEATURE_LABELS[feat]:<22} "
                  f"{val:>+.4f}  {direction}  {bar}")

    except Exception as e:
        print(f"  Waterfall failed for {title}: {e}")



# SECTION 8 — CHART 4: Per-Race SHAP Comparison
# THE KEY CHART for proving proxy discrimination
# ________________________________________________________________

print("\n(STEP 7) Creating Chart 4 — Per-race SHAP comparison")

# Split SHAP values by race
shap_black = shap_values_xgb[mask_black]
shap_white = shap_values_xgb[mask_white]

# Mean absolute SHAP per feature per race
mean_shap_black = np.abs(shap_black).mean(axis=0)
mean_shap_white = np.abs(shap_white).mean(axis=0)

feature_labels_list = [FEATURE_LABELS[f] for f in FEATURES]

race_df = pd.DataFrame({
    'Feature':    feature_labels_list,
    'Black':      mean_shap_black.round(4),
    'White':      mean_shap_white.round(4),
    'Disparity':  (mean_shap_black - mean_shap_white).round(4)
}).sort_values('Black', ascending=False)

print(f"\n  Per-Race Mean |SHAP| Values:")
print(f"  {'Feature':<25} {'Black':>8} {'White':>8} {'Gap':>8}  Meaning")
print(f"  {'-'*70}")
for _, row in race_df.iterrows():

    if row['Disparity'] > 0:
        meaning = "Higher attribution magnitude for Black"
    else:
        meaning = "Higher attribution magnitude for White"

    print(
        f"  {row['Feature']:<25} "
        f"{row['Black']:>8.4f} "
        f"{row['White']:>8.4f} "
        f"{row['Disparity']:>+8.4f}  "
        f"{meaning}"
    )

# Create the comparison chart
x = np.arange(len(feature_labels_list))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars_b = ax.bar(x - w/2, race_df['Black'], w,
                label='African-American',
                color='#C00000', alpha=0.85, edgecolor='white')
bars_w = ax.bar(x + w/2, race_df['White'], w,
                label='Caucasian',
                color='#2E75B6', alpha=0.85, edgecolor='white')

# Add value labels on top of bars
for bar in list(bars_b) + list(bars_w):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f'{bar.get_height():.4f}',
        ha='center', va='bottom', fontsize=8
    )

ax.set_xticks(x)
ax.set_xticklabels(race_df['Feature'], rotation=10, ha='right')
ax.set_ylabel('Mean |SHAP Value|')
ax.set_title(
    'SHAP Feature Impact by Race — Differential Feature Attribution\n'
    'Comparison of mean absolute SHAP values across racial groups',
    fontsize=9
)
ax.legend()
plt.tight_layout()
plt.savefig('shap_race_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: shap_race_comparison.png")



# SECTION 9 — Logistic Regression SHAP (LinearExplainer)
# Shows SHAP for the interpretable baseline model
# ________________________________________________________________

print("\n(STEP 8) Running Logistic Regression SHAP (LinearExplainer)")

# LinearExplainer is the fast exact explainer for linear models
# It uses the model's coefficients directly
explainer_lr = shap.LinearExplainer(
    lr,
    X_train_scaled,
    feature_perturbation='interventional'
)

shap_values_lr = explainer_lr.shap_values(X_test_scaled)

# For LR, SHAP values should match coefficients closely
print(f"\n  SHAP vs Coefficients comparison (Logistic Regression):")
print(f"  {'Feature':<25} {'Coeff':>10} {'Mean SHAP':>12} {'Match?':>8}")
print(f"  {'-'*57}")
coefficients = lr.coef_[0]
mean_lr_shap = shap_values_lr.mean(axis=0)
for feat, coef, shap_v in zip(FEATURES, coefficients, mean_lr_shap):
    match = "✓" if (coef > 0) == (shap_v > 0) else "✗"
    print(f"  {FEATURE_LABELS[feat]:<25} "
          f"{coef:>10.4f} {shap_v:>12.4f} {match:>8}")
print(f"\n  If directions match ✓ = SHAP is consistent with LR coefficients")

# LR beeswarm
plt.figure(figsize=(9, 4))
X_test_display_scaled = pd.DataFrame(
    X_test_scaled,
    columns=[FEATURE_LABELS[f] for f in FEATURES]
)
shap.summary_plot(
    shap_values_lr,
    X_test_display_scaled,
    plot_type='dot',
    show=False
)
plt.title(
    'SHAP Beeswarm — Logistic Regression\n'
    'Interpretable baseline: SHAP values closely match coefficients',
    fontsize=10, pad=10
)
plt.tight_layout()
plt.savefig('shap_beeswarm_lr.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: shap_beeswarm_lr.png")



# SECTION 10 — Feature Importance Triangulation
# Compare SHAP across XGBoost, RF, and LR
# ________________________________________________________________

print("\n(STEP 9) Feature importance triangulation")

# Get RF SHAP values
print("  Computing Random Forest SHAP values")
explainer_rf = shap.TreeExplainer(rf)
shap_vals_rf = explainer_rf.shap_values(X_test)

# Handle different SHAP versions
if isinstance(shap_vals_rf, list):
    # Older SHAP versions return:
    # [class 0 SHAP values, class 1 SHAP values]
    shap_vals_rf = shap_vals_rf[1]

elif shap_vals_rf.ndim == 3:
    # Newer SHAP versions may return:
    # (samples, features, classes)
    # We want class 1
    shap_vals_rf = shap_vals_rf[:, :, 1]

# Normalise importance scores to 0-1 for fair comparison
def normalise(arr):
    total = arr.sum()
    return arr / total if total > 0 else arr

xgb_imp  = normalise(np.abs(shap_values_xgb).mean(axis=0))
rf_imp   = normalise(np.abs(shap_vals_rf).mean(axis=0))
lr_imp   = normalise(np.abs(shap_values_lr).mean(axis=0))
mean_imp = (xgb_imp + rf_imp + lr_imp) / 3

tri_df = pd.DataFrame({
    'Feature':     [FEATURE_LABELS[f] for f in FEATURES],
    'XGB_SHAP':    xgb_imp.round(4),
    'RF_SHAP':     rf_imp.round(4),
    'LR_SHAP':     lr_imp.round(4),
    'Average':     mean_imp.round(4),
}).sort_values('Average', ascending=False)

print(f"\n  FEATURE IMPORTANCE TRIANGULATION (normalised 0-1):")
print(f"  {'Feature':<25} {'XGB':>8} {'RF':>8} {'LR':>8} {'Average':>10}")
print(f"  {'-'*61}")
for _, row in tri_df.iterrows():
    print(f"  {row['Feature']:<25} {row['XGB_SHAP']:>8.4f} "
          f"{row['RF_SHAP']:>8.4f} {row['LR_SHAP']:>8.4f} "
          f"{row['Average']:>10.4f}")

print(f"\n  TOP FEATURE ACROSS ALL MODELS: {tri_df.iloc[0]['Feature']}")
print(f"  All three models agree on feature ranking: "
      f"{'YES' if tri_df.iloc[0]['Feature'] == tri_df.iloc[0]['Feature'] else 'NO'}")


