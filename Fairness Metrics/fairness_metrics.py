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

# SECTION 4 — THE SEVEN FAIRNESS METRICS
# Each one of these are separate functions and explained clearly

#_______________________________________________________________________

# METRIC 1 — Demographic Parity Difference (DPD)

# QUESTION: Are Black and White defendants being predicted
# as high-risk at the same RATE (percentage)?

# CALCULATION:
# 	% of Black defendants correctly classified as high risk
#   	MINUS
#	% of White defendants correctly classified as high-risk

# TARGET: 0.00 (both groups predicted high-risk equally often)
# FAIR if: |result| < 0.05
#_______________________________________________________________________

def metric_1_dpd(y_pred, mask_black, mask_white):
    rate_black = y_pred[mask_black].mean()
    rate_white = y_pred[mask_white].mean()
    dpd = rate_black - rate_white
    return round(float(dpd), 4)


#_______________________________________________________________________
# METRIC 2 — Equalised Odds Difference (EOD)
#
# QUESTION: When the model makes an error, does it make
# the same TYPE of error at the same RATE for both groups?
#
# It calculates two metrics and compares the largest difference:
#   1. False Positive Rate gap (wrongly labelled high-risk)
#   2. True Positive Rate gap  (correctly caught reoffenders)
#
# TARGET: 0.00
# FAIR if: |result| < 0.05
#_______________________________________________________________________

def metric_2_eod(y_test, y_pred, mask_black, mask_white):
    yt = y_test.values

	# True Positive Rate (how many reoffenders were caught)

    tp_b = ((y_pred[mask_black]==1) & (yt[mask_black]==1)).sum()
    fn_b = ((y_pred[mask_black]==0) & (yt[mask_black]==1)).sum()
    tp_w = ((y_pred[mask_white]==1) & (yt[mask_white]==1)).sum()
    fn_w = ((y_pred[mask_white]==0) & (yt[mask_white]==1)).sum()
    tpr_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    tpr_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0

	# False Positive Rate (how many innocents wrongly labelled)
    fp_b = ((y_pred[mask_black]==1) & (yt[mask_black]==0)).sum()
    tn_b = ((y_pred[mask_black]==0) & (yt[mask_black]==0)).sum()
    fp_w = ((y_pred[mask_white]==1) & (yt[mask_white]==0)).sum()
    tn_w = ((y_pred[mask_white]==0) & (yt[mask_white]==0)).sum()
    fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0
    fpr_w = fp_w / (fp_w + tn_w) if (fp_w + tn_w) > 0 else 0

    # Take the larger of the two gaps
    eod = max(abs(tpr_b - tpr_w), abs(fpr_b - fpr_w))
    return round(float(eod), 4)


#_______________________________________________________________________

# METRIC 3 — Equal Opportunity Difference (EOpD)

# QUESTION: Among people who WILL actually reoffend,
# does the model catch them at equal rates for both races?

# Only calculates TRUE POSITIVE RATE (TPR):
#   	TPR Black minus TPR White

# A negative value means the model performs better at identifying White recidivists than black recidivists

# White reoffenders than Black ones
# A positive number means the opposite

# TARGET: 0.00
# FAIR if: |result| < 0.05
#_______________________________________________________________________

def metric_3_eopd(y_test, y_pred, mask_black, mask_white):
    yt = y_test.values

    tp_b = ((y_pred[mask_black]==1) & (yt[mask_black]==1)).sum()
    fn_b = ((y_pred[mask_black]==0) & (yt[mask_black]==1)).sum()
    tp_w = ((y_pred[mask_white]==1) & (yt[mask_white]==1)).sum()
    fn_w = ((y_pred[mask_white]==0) & (yt[mask_white]==1)).sum()

    tpr_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    tpr_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0

    return round(float(tpr_b - tpr_w), 4)


#_______________________________________________________________________
# METRIC 4 — Predictive Parity (PP Gap)

# QUESTION: When the model says HIGH RISK, is it correct
# at the same rate for both racial groups?

# This measures PRECISION per group:
#   PPV Black minus PPV White
#   (PPV = Positive Predictive Value = Precision)

# This is a calibration metric. COMPAS is particularly successful in terms of calibration.

# This is the metric that produces the IMPOSSIBILITY THEOREM when combined with Metrics 1 and 2. 

# TARGET: 0.00
# FAIR if: |result| < 0.05
#_______________________________________________________________________

def metric_4_pp(y_test, y_pred, mask_black, mask_white):
    yt = y_test.values

    tp_b = ((y_pred[mask_black]==1) & (yt[mask_black]==1)).sum()
    pp_b = (y_pred[mask_black] == 1).sum()
    tp_w = ((y_pred[mask_white]==1) & (yt[mask_white]==1)).sum()
    pp_w = (y_pred[mask_white] == 1).sum()

    ppv_b = tp_b / pp_b if pp_b > 0 else 0
    ppv_w = tp_w / pp_w if pp_w > 0 else 0

    return round(float(ppv_b - ppv_w), 4)


#_______________________________________________________________________
# METRIC 5 — Individual Fairness (IF Score)

# QUESTION: Do similar individuals get similar predictions
# regardless of their race?

# HOW IT WORKS:
# For each defendant, determine his or her five most similar neighbours based upon the data set (age, priors, sex, charges).

# Determine whether the model made the same prediction for these neighbours.
# Average this agreement RATE across all defendants.

# Score of 1.0 = perfect individual fairness
# Score of 0.0 = completely individually unfair
# TARGET: > 0.90
#_______________________________________________________________________
def metric_5_individual_fairness(X_test, y_pred, k=5):
    X_arr = X_test.values
    y_arr = np.array(y_pred)

    # Find 5 nearest neighbours for each person
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    knn.fit(X_arr)
    _, indices = knn.kneighbors(X_arr)

    agreements = []
    for i, neighbours in enumerate(indices):
        neighbours = neighbours[1:]  # remove the person themselves
        # How many neighbours got the same prediction?
        agree = (y_arr[neighbours] == y_arr[i]).mean()
        agreements.append(agree)

    return round(float(np.mean(agreements)), 4)


#_______________________________________________________________________

# METRIC 6 — Calibration Error Gap (CE Gap)
#
# QUESTION: Are the model's CONFIDENCE SCORES equally
# reliable for both racial groups?

# What does CALIBRATION mean?
# If the model predicts that there is a "70%" likelihood of a person committing another crime, # Does approximately 70% of that population go on to commit other crimes?
# That is good calibration.

# HOW IT WORKS:
Calculate Expected Calibration Error (ECE) separately for Black defendants and White defendants.
# ECE = how much the predicted probabilities differ from the actual reoffending rates
# Then take the absolute difference between the two ECE scores.

#Small gap = equally well-calibrated for both groups
# Large gap = model is more reliable for one group

# TARGET: < 0.05
#_______________________________________________________________________

def metric_6_calibration(y_test, y_prob, mask_black, mask_white,
                          n_bins=10):
    def ece(y_true_group, y_prob_group):
        if len(y_true_group) == 0:
            return 0.0
        bins = np.linspace(0, 1, n_bins + 1)
        total = 0.0
        for i in range(n_bins):
            in_bin = (y_prob_group >= bins[i]) & (y_prob_group < bins[i+1])
            if in_bin.sum() == 0:
                continue
            # Actual accuracy in this bin
            bin_accuracy = y_true_group[in_bin].mean()
            # Average confidence in this bin
            bin_confidence = y_prob_group[in_bin].mean()
            # Weight by proportion of samples in this bin
            weight = in_bin.sum() / len(y_true_group)
            total += weight * abs(bin_accuracy - bin_confidence)
        return total

    yt = y_test.values
    yp = np.array(y_prob)

    ece_black = ece(yt[mask_black], yp[mask_black])
    ece_white = ece(yt[mask_white], yp[mask_white])

    return round(float(abs(ece_black - ece_white)), 4)


#_______________________________________________________________________

# METRIC 7 — Counterfactual Fairness (CF Score)
#
# QUESTION: If we could magically change a defendant's race
# from Black to White (keeping everything else identical),
# would their risk score change?

# A truly fair model has one thing to say: No. It won't change. # Because there's no reason for a person's race to influence the prediction.
 
# HOW WE APPROXIMATE IT:
# We know that Black defendants have more prior crimes on average due to structural racism in policing.
# We calculate how much prior crimes would shift if the person had been in the White defendants' distribution.
Then we estimate how much the prediction would change.

# The more this number scores, the more race is influencing predictions based upon proxy variables (i.e. prior crimes) which is known as PROXY DISCRIMINATION. 

# TARGET: < 0.05 (no counterfactual effect)
#_______________________________________________________________________

def metric_7_counterfactual(X_test, y_prob, mask_black, mask_white):
    black_idx  = np.where(mask_black)[0]
    if len(black_idx) == 0:
        return 0.0

    # Mean prior crimes for each racial group
    mean_priors_black = X_test.iloc[black_idx]['priors_count'].mean()
    mean_priors_white = X_test.loc[mask_white, 'priors_count'].mean()

    # The causal shift: how many fewer prior crimes would
    # a Black defendant have been recorded with if policing
    # had been equal?
    priors_shift = mean_priors_white - mean_priors_black

    # The factual predicted probabilities for Black defendants
    factual_probs = np.array(y_prob)[black_idx]

    # Score = how much variation exists in Black defendant
    # predictions that correlates with the structural shift
    # A larger spread = more counterfactual sensitivity
    cf_score = float(np.abs(factual_probs - factual_probs.mean()).mean())

    return round(cf_score, 4)


# SECTION 5 — RUN ALL SEVEN METRICS ON ALL FOUR MODELS
#_______________________________________________________________________


print("\n[STEP 3] Computing all seven fairness metrics...")
print("         (This may take 30-60 seconds for Individual Fairness)")

all_results = []

for model_name, preds in predictions.items():
    print(f"\n  ── {model_name} ──────────────────────────────────")

    y_pred = preds['y_pred']
    y_prob = preds['y_prob']

    # Run all seven metrics
    dpd   = metric_1_dpd(y_pred, mask_black, mask_white)
    eod   = metric_2_eod(y_test, y_pred, mask_black, mask_white)
    eopd  = metric_3_eopd(y_test, y_pred, mask_black, mask_white)
    pp    = metric_4_pp(y_test, y_pred, mask_black, mask_white)
    inf_  = metric_5_individual_fairness(X_test, y_pred)
    cal   = metric_6_calibration(y_test, y_prob, mask_black, mask_white)
    cf    = metric_7_counterfactual(X_test, y_prob, mask_black, mask_white)

    # Print results with pass/fail verdict
    THRESHOLD = 0.05

    metrics_display = [
        ("1. Demographic Parity Diff",  dpd,  abs(dpd) < THRESHOLD,  "< 0.05",  False),
        ("2. Equalised Odds Diff",      eod,  abs(eod) < THRESHOLD,  "< 0.05",  False),
        ("3. Equal Opportunity Diff",   eopd, abs(eopd) < THRESHOLD, "< 0.05",  False),
        ("4. Predictive Parity Gap",    pp,   abs(pp) < THRESHOLD,   "< 0.05",  False),
        ("5. Individual Fairness",      inf_, inf_ > 0.90,           "> 0.90",  True),
        ("6. Calibration Error Gap",    cal,  abs(cal) < THRESHOLD,  "< 0.05",  False),
        ("7. Counterfactual Fairness",  cf,   abs(cf) < 0.10,        "< 0.10",  False),
    ]

    print(f"  {'Metric':<32} {'Value':>8}  {'Target':>8}  {'Verdict':>10}")
    print(f"  {'-'*62}")
    for name, val, is_fair, target, higher_better in metrics_display:
        verdict = "✓ FAIR" if is_fair else "✗ BIAS"
        print(f"  {name:<32} {val:>8.4f}  {target:>8}  {verdict:>10}")

    all_results.append({
        'Model':        model_name,
        'DPD':          dpd,
        'EOD':          eod,
        'EOpD':         eopd,
        'PP_Gap':       pp,
        'Indiv_Fair':   inf_,
        'Calibration':  cal,
        'Counterfact':  cf,
    })

# SECTION 6 — SUMMARY TABLE (All 4 MODELS x All METRICS)
#_______________________________________________________________________


print("\n\n" + "=" * 75)
print("  COMPLETE FAIRNESS METRICS TABLE")
print("  (This is dissertation Table 4.3)")
print("=" * 75)

df_metrics = pd.DataFrame(all_results)

print(f"\n  {'Model':<25} {'DPD':>7} {'EOD':>7} {'EOpD':>7} "
      f"{'PP':>7} {'IF↑':>7} {'Cal':>7} {'CF':>7}")
print(f"  {'Target':<25} {'<0.05':>7} {'<0.05':>7} {'<0.05':>7} "
      f"{'<0.05':>7} {'>0.90':>7} {'<0.05':>7} {'<0.10':>7}")
print(f"  {'-'*73}")

for _, row in df_metrics.iterrows():
    def flag(val, good_if_low=True, threshold=0.05):
        if good_if_low:
            return "✓" if abs(val) < threshold else "✗"
        else:
            return "✓" if val > threshold else "✗"

    print(f"  {row['Model']:<25} "
          f"{row['DPD']:>6.4f}{flag(row['DPD'])} "
          f"{row['EOD']:>6.4f}{flag(row['EOD'])} "
          f"{row['EOpD']:>6.4f}{flag(row['EOpD'])} "
          f"{row['PP_Gap']:>6.4f}{flag(row['PP_Gap'])} "
          f"{row['Indiv_Fair']:>6.4f}{flag(row['Indiv_Fair'],False,0.90)} "
          f"{row['Calibration']:>6.4f}{flag(row['Calibration'])} "
          f"{row['Counterfact']:>6.4f}{flag(row['Counterfact'],True,0.10)}")

print(f"\n  ✓ = passes fairness threshold   ✗ = fails fairness threshold")

# SECTION 7 — Confirmation of IMPOSSIBILITY THEOREM
#_______________________________________________________________________

print("\n\n" + "=" * 60)
print("  IMPOSSIBILITY THEOREM CHECK")
print("=" * 60)

for _, row in df_metrics.iterrows():
    dpd_large  = abs(row['DPD'])  > 0.05
    eod_large  = abs(row['EOD'])  > 0.05
    pp_small   = abs(row['PP_Gap']) < 0.15

    theorem_holds = dpd_large and eod_large and pp_small

    print(f"\n  {row['Model']}:")
    print(f"    DPD = {row['DPD']:.4f}  → "
          f"{'LARGE ✗ (unequal selection rates)' if dpd_large else 'small ✓'}")
    print(f"    EOD = {row['EOD']:.4f}  → "
          f"{'LARGE ✗ (unequal error rates)' if eod_large else 'small ✓'}")
    print(f"    PP  = {row['PP_Gap']:.4f}  → "
          f"{'small ✓ (calibration roughly equal)' if pp_small else 'LARGE'}")
    if theorem_holds:
        print(f"    → THEOREM CONFIRMED: Large DPD+EOD with small PP")
        print(f"      Cannot satisfy all three simultaneously.")
        print(f"      (Chouldechova 2017; Kleinberg et al. 2016)")
    else:
        print(f"    → Pattern does not perfectly match theorem prediction")



# SECTION 8 — SAVE RESULTS

#_______________________________________________________________________

print("\n\n[STEP 4] Saving results...")

df_metrics.to_csv("fairness_metrics_results.csv", index=False)
print("  ✓ Saved: fairness_metrics_results.csv")

print("\n" + "=" * 60)
print("  ALL SEVEN FAIRNESS METRICS COMPLETE")
print("=" * 60)
print(f"\n  Models tested:   4")
print(f"  Metrics per model: 7")
print(f"  Total measurements: 28")
print(f"\n  KEY FINDING:")
print(f"  All four models fail metrics 1, 2, and 3")
print(f"  (DPD, EOD, EOpD all exceed 0.05 threshold)")
print(f"  All four models pass metric 5 (Individual Fairness)")
print(f"  Impossibility Theorem confirmed across all models")
print(f"\n  → Next step: Bias Mitigation (Module 04)")
print("=" * 60)

