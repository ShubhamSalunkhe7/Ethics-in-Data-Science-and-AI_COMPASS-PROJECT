
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




