from pathlib import Path
import pandas as pd
import numpy as np


data=pd.read_parquet("data/prediction_ticks.parquet")

print(data.shape)
data.describe()
print(data.columns)

###################################  Start with Logistic Regression ####################################
#
#1 Split your data into train and test sets, keeping the result column hidden from the model during prediction, as you described.
#2 Fit logistic regression as a baseline.
#3 Fit LightGBM or random forest and compare using cross-validation, not just one split.
#4 Choose your metric carefully. If Yes/No is imbalanced (say 95% No), accuracy is misleading. Use F1, precision/recall, or ROC-AUC instead.
#5 Tune only the winner, and evaluate it once on the held-out test set.