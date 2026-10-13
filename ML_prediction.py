import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, log_loss
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot as plt

SEED = 1111

def init():
    data=pd.read_csv("data/adjusted_prediction_dataset.csv").dropna()
    data.drop('Unnamed: 0', axis=1, inplace=True)
    id_cols = ["kill_id", "A_steamid", "B_steamid", "match_name","A_name","B_name"]
    label_col = "A_dies"
    feature_cols = [c for c in data.columns if c not in ( id_cols + [label_col] )]
    X = data[feature_cols]
    y = data[label_col]
    groups = data["match_name"]
    print("Featured Columns: ", feature_cols)
    return X, y, groups

'''Print a report of the model's performance'''
def reporting(test_proba, y_test):
    print("Log loss:", log_loss(y_test, test_proba))
    print("Accuracy:", accuracy_score(y_test, test_proba > 0.5))
    print("ROC_AUC:", roc_auc_score(y_test, test_proba))
    # print(classification_report(y_test, test_proba > 0.5))
    RocCurveDisplay.from_predictions(y_test, test_proba)
    plt.show()


def split_data(X, y, groups):
    # Split 1: separate out the test set by match
    test_set = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=SEED)
    trainval_idx, test_idx = next(test_set.split(X, y, groups))

    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]

    print(f"train: {len(X_trainval)}  test: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        # min_samples_leaf=10,
        min_samples_split=5,  # guards against overfitting on noisy per-tick features
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced"  # in case A_dies isn't ~50/50 after dropping failed merges etc.
    )
    print(clf)

    clf.fit(X_trainval, y_trainval)
    test_proba = clf.predict_proba(X_test)[:, 1]
    reporting(test_proba,y_test)


def split_data_gb(X, y, groups):
    # Split 1: separate out the test set by match
    test_set = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=SEED)
    trainval_idx, test_idx = next(test_set.split(X, y, groups))

    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]

    print()
    print(f"train: {len(X_trainval)}   test: {len(X_test)}")

    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        subsample=.5,
        random_state=SEED,
    )
    print(clf)

    clf.fit(X_trainval, y_trainval)
    test_proba = clf.predict_proba(X_test)[:, 1]
    reporting(test_proba, y_test)


    return clf



X,y,groups = init()
split_data(X,y,groups)
split_data_gb(X,y,groups)

# Log loss explanation :
# >0.693	No better than guessing 50/50
# 0.5–0.65	Weak but real signal
# 0.3–0.5	Decent, reasonably confident and mostly right
# < 0.3	Strong, well-calibrated model
# near 0	Suspiciously perfect


# List of remaining columns to add to dataset :
# Move Speed
# Z difference
# HP/armor/equip differences
# X,Y difference
# Flash/smoke status
# Round context : 5v5; 3v2
# Is there multiple enemies on screen
# Is the duel happening after a kill (trade potential)

# Faire d'abord extract kills puis ensuite parse les ticks pour avoir les sample ticks au lieu d'avoir les ticks au moment du kills