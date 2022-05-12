import os
from aif360.sklearn.datasets import fetch_adult, fetch_german
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
from aif360.datasets import AdultDataset, BinaryLabelDataset, BankDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
)
from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    average_odds_error,
    generalized_fpr,
)

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
import pandas as pd
from pandas.api.types import is_numeric_dtype
from aif360.metrics import BinaryLabelDatasetMetric
from train_model import fair_models
from scipy import stats

if os.path.isdir("/home/sahil/ML/Results") == False:
    os.mkdir("Results")

os.chdir("/home/sahil/ML/Results")


def fair_acc_test(train_obj, cross_val):
    cross_val["C"] = cross_val["C"].astype(float)
    cross_val = cross_val.sort_values(by="C", ascending=False)
    cross_val["eq"] = 1 - abs(cross_val["Equal Opportunity Difference"])
    cross_val["score"] = stats.hmean(cross_val[["Average CV Accuracy", "eq"]], axis=1)
    # cross_val["change_2"] = abs(cross_val.change.diff())

    best_c_value = float(cross_val.at[cross_val.score.idxmax(), "C"])
    print(best_c_value)
    lrobj = LogisticRegression(C=best_c_value, solver="liblinear")
    x_train = train_obj.train.features
    y_train = train_obj.train.labels
    x_test = train_obj.test.features
    y_test = train_obj.test.labels
    lrobj.fit(x_train, y_train)
    lr_fairacc_wise = {
        "Accuracy": [lrobj.score(x_test, y_test)],
        "Equal Opportunity Difference": [
            train_obj.test_metric(train_obj.test, lrobj.predict(x_test))
        ],
    }
    return pd.DataFrame(lr_fairacc_wise), best_c_value, cross_val

    # Calculate fairness+accuracy score


best_c_values = {}
bank = BankDataset()
data = load_preproc_data_adult(["sex"])
unprivileged_groups_adult = [{"sex": 0}]
privileged_groups_adult = [{"sex": 1}]
unprivileged_groups_bank = [{"age": 0}]
privileged_groups_bank = [{"age": 1}]
weighted_train_adult = fair_models(
    unprivileged_groups=unprivileged_groups_adult,
    privileged_groups=privileged_groups_adult,
    data=data,
    reweighing=True,
    suffix="adult_notdropped",
)

cross_val1, test1 = weighted_train_adult.get_results()
res, best_c, cv1 = fair_acc_test(weighted_train_adult, cross_val1)
best_c_values["adult_notdropped_weighted"] = best_c
print(res)
res.to_pickle("test_weighted_train.pkl")
unweighted_train_adult = fair_models(
    unprivileged_groups=unprivileged_groups_adult,
    privileged_groups=privileged_groups_adult,
    data=data,
    reweighing=False,
    suffix="adult_notdropped",
)


cross_val1, test1 = unweighted_train_adult.get_results()
res, best_c, cv2 = fair_acc_test(unweighted_train_adult, cross_val1)
best_c_values["adult_notdropped_unweighted"] = best_c
res.to_pickle("test_unweighted_train.pkl")

train_adult_drop = fair_models(
    unprivileged_groups=unprivileged_groups_adult,
    privileged_groups=privileged_groups_adult,
    data=data,
    reweighing=False,
    suffix="adult_dropped",
    drop_prot_feats=True,
)
best_c_values["adult_dropped"] = best_c

cross_val1, test1 = train_adult_drop.get_results()
res, best_c, cv3 = fair_acc_test(train_adult_drop, cross_val1)
res.to_pickle("test_adult_drop.pkl")

weighted_train_bank = fair_models(
    unprivileged_groups=unprivileged_groups_bank,
    privileged_groups=privileged_groups_bank,
    data=bank,
    reweighing=True,
    suffix="bank_notdropped",
)

cross_val1, test1 = weighted_train_bank.get_results()
res, best_c, cv4 = fair_acc_test(weighted_train_bank, cross_val1)
best_c_values["bank_notdropped_weighted"] = best_c
res.to_pickle("test_bank_notdrop_weighted.pkl")
print(res)

unweighted_train_bank = fair_models(
    unprivileged_groups=unprivileged_groups_bank,
    privileged_groups=privileged_groups_bank,
    data=bank,
    reweighing=False,
    suffix="bank_notdropped",
)

cross_val2, test2 = unweighted_train_bank.get_results()
res, best_c, cv5 = fair_acc_test(unweighted_train_bank, cross_val2)
best_c_values["bank_notdropped_unweighted"] = best_c
res.to_pickle("test_bank_notdrop_unweighted.pkl")
print(res)

drop_train_bank = fair_models(
    unprivileged_groups=unprivileged_groups_bank,
    privileged_groups=privileged_groups_bank,
    data=bank,
    reweighing=False,
    suffix="bank_dropped",
)

cross_val3, test3 = drop_train_bank.get_results()
res, best_c, cv6 = fair_acc_test(drop_train_bank, cross_val3)
best_c_values["bank_dropped"] = best_c
res.to_pickle("test_bank_dropped.pkl")
print(res)


print(cross_val1)
print(res)


def test_adult():
    unprivileged_groups = [{"sex": 0}]
    privileged_groups = [{"sex": 1}]
    dataset_orig = load_preproc_data_adult(["sex"])
    weighted_train = fair_models(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        data=dataset_orig,
        reweighing=True,
        suffix="Adult",
    )
    unweighted_train = fair_models(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        data=dataset_orig,
        reweighing=False,
        suffix="Adult",
    )

    print("Without reweighing Results")
    cross_val, test = unweighted_train.get_results()
    print(cross_val)
    print("With reweighing Results")
    cross_val, test = weighted_train.get_results()
    print(cross_val)


# test_adult()
