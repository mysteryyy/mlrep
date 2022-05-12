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


def eq_odd_metric(ytrue, ypred, A):
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


def pareto_curve(
    dataset,
    figname,
    gamma_list=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25],
):

    # tree_predictor = tree.DecisionTreeRegressor(max_depth=3)

    results_dict = {}
    max_iters = 150
    fair_clf = GerryFairClassifier(
        C=10,
        printflag=True,
        gamma=1,
        max_iters=max_iters,
    )
    fair_clf.printflag = False
    fair_clf.max_iters = max_iters
    errors, fp_violations, fn_violations = fair_clf.pareto(dataset, gamma_list)
    results_dict = {
        "gamma": gamma_list,
        "errors": errors,
        "fp_violations": fp_violations,
        "fn_violations": fn_violations,
    }
    plt.plot(errors, fp_violations)
    plt.xlabel("Error")
    plt.ylabel("Unfairness")
    plt.legend()
    plt.title(f"Error vs. Unfairness\n({figname} Dataset)")
    plt.savefig(f"gerryfair_pareto_{figname}.png")
    plt.close()
    plt.show()
    return results_dict


def group_preds_demo(train, figname, gamma):
    results = pareto_curve(train, figname)
    res_fair = pd.DataFrame(results)
    res_fair["score"] = stats.hmean(res_fair[["errors", "fp_violations"]], axis=1)
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
    tr = train.convert_to_dataframe()[0]
    tr["predictions"] = dt_yhat.labels.ravel()
    acc = accuracy_score(np.array(tr.labels), np.array(tr.predictions))
    gerry_metric = BinaryLabelDatasetMetric(test)
    gamma_disparity = gerry_metric.rich_subgroup(array_to_tuple(dt_yhat.labels), "FP")
    print("Gamma Disparity: ", gamma_disparity)
    print("Accuracy: ", acc)
    return acc, gamma_disparity


def make_score(res):
    res["inv violate"] = 1 - res.fp_violations
    res["inv error"] = 1 - res.errors
    res["score"] = stats.hmean(res[["inv violate", "inv error"]], axis=1)
    return res


train, test = bank_preproc()
results = pareto_curve(train, "bank")
res = pd.DataFrame(results)
res = make_score(res)
res.to_pickle("bank_pareto.pkl")
# group_preds_demo(train, "bank", 0.02)
# fair_clf = GerryFairClassifier(C=10, printflag=True, gamma=0.02, max_iters=50)
# clf.fit(train)
train, test = adult_preproc()
results_adult = pareto_curve(train, "adult")
results_adult = pd.DataFrame(results_adult)
resad = make_score(results_adult)
resad.to_pickle("adults_pareto.pkl")
# group_preds_demo(train, "adult", 0.02)
print(results_adult)


# print(gamma)
# print(res_fair)
