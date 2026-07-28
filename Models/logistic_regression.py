
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

