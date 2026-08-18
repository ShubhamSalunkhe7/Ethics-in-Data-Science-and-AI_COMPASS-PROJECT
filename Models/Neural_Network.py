
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




