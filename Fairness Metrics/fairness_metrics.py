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

