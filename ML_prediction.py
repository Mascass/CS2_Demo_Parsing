import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def init():
    data=pd.read_csv("data/prediction_dataset.csv")
    id_cols = ["kill_id", "A_steamid", "B_steamid", "match_name"]
    label_col = "A_dies"
    feature_cols = [c for c in data.columns if c not in ( id_cols + [label_col] + ["A_name","B_name"])]

    X = data[feature_cols]
    y = data[label_col]
    groups = data["match_name"]

    return X, y, groups

def split_data(X, y, groups):
    # Split 1: separate out the test set by match
    test_set = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=111)
    trainval_idx, test_idx = next(test_set.split(X, y, groups))

    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]
    groups_trainval = groups.iloc[trainval_idx]

    # Split 2: carve validation out of what's left, still by match
    validation_set = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=111)
    train_idx, val_idx = next(validation_set.split(X_trainval, y_trainval, groups_trainval))

    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    print(f"train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,  # guards against overfitting on noisy per-tick features
        random_state=111,
        n_jobs=-1,
        class_weight="balanced"  # in case A_dies isn't ~50/50 after dropping failed merges etc.
    )
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)[:, 1]
    print("Validation accuracy:", accuracy_score(y_val, val_pred))
    print("Validation AUC:", roc_auc_score(y_val, val_proba))
    print(classification_report(y_val, val_pred))


    test_pred = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)[:, 1]
    print("Test accuracy:", accuracy_score(y_test, test_pred))
    print("Test AUC:", roc_auc_score(y_test, test_proba))

#
# X,y,groups = init()
# split_data(X,y,groups)

df=pd.read_csv("data/prediction_dataset.csv")
print(df.columns)

###################################  Start with Logistic Regression ####################################
#
#1 Split your data into train and test sets, keeping the result column hidden from the model during prediction, as you described.
#2 Fit logistic regression as a baseline.
#3 Fit LightGBM or random forest and compare using cross-validation, not just one split.
#4 Choose your metric carefully. If Yes/No is imbalanced (say 95% No), accuracy is misleading. Use F1, precision/recall, or ROC-AUC instead.
#5 Tune only the winner, and evaluate it once on the held-out test set.