import pandas as pd
import warnings
import numpy as np
from sklego.preprocessing import InformationFilter
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
)
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.sklearn.datasets import fetch_adult, fetch_german, fetch_bank
from sklego.linear_model import DemographicParityClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer
from sklego.metrics import correlation_score


def bank_preproc():
    # function for cleaning up
    x, y = fetch_bank()
    x = x.drop(columns=["poutcome"])
    x = x.dropna()
    x = pd.get_dummies(x)
    x["target"] = y[x.index]
    x["target"] = x.target.apply(lambda x: 1 if x == "yes" else 0)
    x.index = x["age"]
    data = BinaryLabelDataset(
        df=x, label_names=["target"], protected_attribute_names=["age"]
    )
    np.random.seed(234)
    train, test = data.split([0.7], shuffle=True)
    return train, test


def adult_preproc():
    x, y, sample_weight = fetch_adult()
    x.index = x["age"]
    y.index = x["age"]
    x = x[["age", "education-num", "capital-gain", "hours-per-week", "capital-loss"]]
    x["label"] = y
    x["target"] = x.label.apply(lambda x: 1 if x == ">50K" else 0)
    x = x.drop(["label"], axis=1)
    data = BinaryLabelDataset(
        df=x, label_names=["target"], protected_attribute_names=["age"]
    )
    np.random.seed(2025)
    train, test = data.split([0.7], shuffle=True)
    return train, test


def eq_opp_metric(ytrue, ypred, A):
    df = pd.DataFrame()
    df["label"] = ytrue
    df["predicted"] = ypred
    df["group"] = A
    size_1 = len(df[df.group == 1])
    size_0 = len(df[df.group == 0])

    return (
        len(df[df.group == 1][df.label == df.predicted]) / size_1
        - len(df[df.group == 0][df.label == df.predicted]) / size_0
    )


np.random.seed(365)
train, test = bank_preproc()
train_df, test_df = train.convert_to_dataframe()[0], test.convert_to_dataframe()[0]
x_train, y_train = train_df.drop(columns=["target"]), train_df["target"]

x_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

ifilt = InformationFilter(["age"]).fit(x_train)

# Input features with correlation from age removed
x_train_trans = ifilt.transform(x_train)
x_test_trans = ifilt.transform(x_test)

x_drop_train = x_train.drop(columns=["age"])

x_drop_test = x_test.drop(columns=["age"])

lr_bias = LogisticRegression(solver="liblinear")

lr_bias.fit(x_drop_train, y_train)

pred_train_bias = lr_bias.predict_proba(x_drop_train)[:, 0]


pred_test_bias = lr_bias.predict_proba(x_drop_test)[:, 0]


age_train = np.array(x_train["age"])

age_test = np.array(x_test["age"])

np.random.seed(7)
fair_clf = GridSearchCV(
    estimator=DemographicParityClassifier(
        sensitive_cols="age", covariance_threshold=0.5
    ),
    param_grid={"estimator__covariance_threshold": np.linspace(0.01, 1.00, 5)},
    cv=5,
    refit="accuracy_score",
    return_train_score=True,
    scoring={
        "correlation_score": correlation_score("age"),
        "accuracy_score": make_scorer(accuracy_score),
    },
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fair_clf.fit(x_train, y_train)

    pltr = pd.DataFrame(fair_clf.cv_results_).set_index(
        "param_estimator__covariance_threshold"
    )


res = pltr[["mean_test_accuracy_score", "mean_test_correlation_score"]]
# Compute score
res["mod_corr"] = 1 - abs(res["mean_test_correlation_score"])
res["score"] = stats.hmean(res[["mod_corr", "mean_test_accuracy_score"]], axis=1)
best_thresh = float(res.score.idxmax())
print("best covariance threshold: ", best_thresh)

final_clf = DemographicParityClassifier(
    sensitive_cols="age", covariance_threshold=best_thresh
)
final_clf.fit(x_train, y_train)
print(res)

print(
    "training set biased correlation: ", np.corrcoef(age_train, pred_train_bias)[0][1]
)

print("Testing set biased correlation: ", np.corrcoef(age_test, pred_test_bias)[0][1])

print(correlation_score("age")(final_clf, x_test, y_test))
