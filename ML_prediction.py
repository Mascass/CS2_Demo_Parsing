import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, log_loss
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot as plt

SEED = 1111
DATA_PATH="data/adjusted_prediction_dataset.csv"

'''Initialize data and groups'''
def init(path=DATA_PATH):
    data=pd.read_csv(path).dropna().reset_index(drop=True)
    data.drop(columns=['Unnamed: 0',"B_CT"], inplace=True)
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
    print("Log loss:", f"{log_loss(y_test, test_proba):.4f}")
    print("Accuracy:", f"{accuracy_score(y_test, test_proba > 0.5):.4f}")
    print("ROC_AUC:", f"{roc_auc_score(y_test, test_proba):.4f}")
    # print(classification_report(y_test, test_proba > 0.5))
    RocCurveDisplay.from_predictions(y_test, test_proba)
    plt.show()

''' returns dict of prediction models '''
def make_models():
    return {
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            # min_samples_leaf=10,
            min_samples_split=5,  # guards against overfitting on noisy per-tick features
            random_state=SEED,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        subsample=.5,
        random_state=SEED,
    ),
    }


def split_data(X, y, groups, clf):
    # Split 1: separate out the test set by match
    test_set = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=SEED)
    trainval_idx, test_idx = next(test_set.split(X, y, groups))

    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]

    print(f"train: {len(X_trainval)}  test: {len(X_test)}")

    print(clf)

    clf.fit(X_trainval, y_trainval)
    test_proba = clf.predict_proba(X_test)[:, 1]
    reporting(test_proba,y_test)


def get_trainval_idx(X, y, groups):
    # Same seed / group split as split_data so the test set stays untouched by CV
    test_set = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=SEED)
    trainval_idx, _ = next(test_set.split(X, y, groups))
    return trainval_idx


def cross_validate_models(X, y, groups, models, n_splits=5):
    # Grouped CV (by match) on the train/val part only. Returns model names ranked best to worst by log loss
    trainval_idx = get_trainval_idx(X, y, groups)
    X_tv, y_tv, g_tv = X.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]

    cv = GroupKFold(n_splits=n_splits)
    scoring = {"log_loss": "neg_log_loss", "accuracy": "accuracy", "roc_auc": "roc_auc"}
    results = {}
    for name, clf in models.items():
        scores = cross_validate(clone(clf), X_tv, y_tv, groups=g_tv, cv=cv, scoring=scoring, n_jobs=1)
        results[name] = {
            "log_loss": (-scores["test_log_loss"].mean(), scores["test_log_loss"].std()),
            "accuracy": (scores["test_accuracy"].mean(), scores["test_accuracy"].std()),
            "roc_auc": (scores["test_roc_auc"].mean(), scores["test_roc_auc"].std()),
        }
        print(f"\n[CV {n_splits}-fold grouped] {name}")
        print("Time elapsed : ",(scores["fit_time"].mean() + scores["score_time"].mean())*n_splits)
        for metric, (m, sd) in results[name].items():
            print(f"  {metric}: {m:.4f} +/- {sd:.4f}")

    return sorted(results, key=lambda n: results[n]["log_loss"][0])


# Init Data
X, y, groups = init()
models = make_models()

# Cross-validation on train/val only
ranking = cross_validate_models(X, y, groups, models)
print()
print("Ranking by Cross Validation log loss:", ranking)

# Final test on held-out matches with the 2 best models
best_two = ranking[:2]
print("Testing: ", best_two)
for name in best_two:
    split_data(X, y, groups, models[name])

# Log loss explanation :
# >0.693	No better than guessing 50/50
# 0.5–0.65	Weak but real signal
# 0.3–0.5	Decent, reasonably confident and mostly right
# < 0.3	Strong, well-calibrated model
# near 0	Suspiciously perfect


# List of remaining columns to add to dataset :
# Move Speed
# X,Y difference
# Flash/smoke status
# Round context : 5v5; 3v2
# Is there multiple enemies on screen
# Is the duel happening after a kill (trade potential)
