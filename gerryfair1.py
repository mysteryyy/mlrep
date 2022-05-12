import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from aif360.algorithms.inprocessing.gerryfair.auditor import *
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from aif360.datasets import AdultDataset, BinaryLabelDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
)
from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    average_odds_error,
    generalized_fpr,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
from aif360.sklearn.datasets import fetch_adult, fetch_german, fetch_bank
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    average_odds_error,
    generalized_fpr,
)
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
from sklearn import tree
from scipy import stats


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


def group_preds_demo(train, figname, gamma):
    # For demonstrating the groups found by the auditor
    fair_clf = GerryFairClassifier(C=10, printflag=True, gamma=gamma, max_iters=2)
    fair_clf.fit(train)
    dt_yhat = fair_clf.predict(train, threshold=False)
    audit = Auditor(train, "FP")  # initialise an auditor class
    # get group labels
    gpred, gdisp = audit.audit(dt_yhat.labels.ravel())
    tr = train.convert_to_dataframe()[0]
    tr["group_label"] = gpred
    tr["predictions"] = dt_yhat.labels.ravel()
    sns.scatterplot(
        y="predictions", x="age", data=tr, hue="group_label"
    ).get_figure().savefig(f"initital_group_predictions_{figname}.png")
    # gamma = res_fair.at[res_fair.score.idxmax(), "C"]


def preds(clf, test):

    dt_yhat = clf.predict(test, threshold=False)
    # get group labels
    acc = accuracy_score(test.labels, dt_yhat.labels)
    gerry_metric = BinaryLabelDatasetMetric(test)
    gamma_disparity = gerry_metric.rich_subgroup(array_to_tuple(dt_yhat.labels), "FP")
    print("Gamma Disparity: ", gamma_disparity)
    print("Accuracy: ", acc)
    return acc, gamma_disparity


train, test = bank_preproc()
group_preds_demo(train, "bank", 0.02)
clf = GerryFairClassifier(
    C=10,
    printflag=True,
    gamma=0.02,
    max_iters=100,
)
clf.fit(train, early_termination=True)

acc, gamma_disp = preds(clf, test)
print("bank test accuracy: ", acc)
print("bank test gamma disparity: ", gamma_disp)
# group_preds_demo(train, "bank", 0.02)
# fair_clf = GerryFairClassifier(C=10, printflag=True, gamma=0.02, max_iters=50)
# clf.fit(train)
train, test = adult_preproc()
group_preds_demo(train, "adult", 0.02)
clf = GerryFairClassifier(
    C=10,
    printflag=True,
    gamma=0.02,
    max_iters=100,
)
clf.fit(train, early_termination=True)
acc, gamma_disp = preds(clf, test)
print("adult test accuracy: ", acc)
print("adult test gamma disparity: ", gamma_disp)
#
