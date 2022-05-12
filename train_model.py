import matplotlib.pyplot as plt
import pickle as pck
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

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
from aif360.datasets import AdultDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
)
from aif360.datasets import StructuredDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    average_odds_error,
    generalized_fpr,
)
import keras
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from typing import Dict, Iterable, Any


# STEP 3: We split between training and test set.

# Normalize the dataset, both train and test. This should always be done in any machine learning pipeline!


class fair_models:
    def __init__(
        self,
        suffix,
        unprivileged_groups,
        privileged_groups,
        data,
        reweighing=False,
        drop_prot_feats=False,
    ):
        np.random.seed(2021)
        self.suffix = suffix
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.train, self.test = data.split([0.7], shuffle=True)
        self.reweighing = reweighing
        self.drop_prot_feats = drop_prot_feats
        if reweighing == True:
            RW = Reweighing(
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups,
            )

            self.train = RW.fit_transform(self.train)

    @staticmethod
    def get_mlp_classes(test, model):

        y_prob = model.predict(test.features)
        y_prob = pd.Series(y_prob.ravel())
        y_prob[y_prob > 0.5] = 1
        y_prob[y_prob < 0.5] = 0
        return np.array(y_prob)

    def remove_protected_feats(self, data, feats_to_drop):

        dt = data.convert_to_dataframe()[0]
        dt = dt.drop(columns=feats_to_drop)
        final_dt = StructuredDataset(
            df=dt, label_names=self.train.label_names, protected_attribute_names=[]
        )
        return final_dt

    def create_model(self, lr, shape):

        # create model
        model = Sequential()
        model.add(Dense(30, input_dim=shape, activation="sigmoid"))
        model.add(Dense(30, activation="sigmoid"))
        model.add(Dense(1, activation="sigmoid"))
        opt = keras.optimizer_v1.adam(lr=lr)
        model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=opt)
        return model

    def train_model(self, train, reg_param):
        x_train = train.features
        if self.drop_prot_feats == False:
            assert x_train.shape[1] == self.train.features.shape[1]
        y_train = train.labels.ravel()
        # if model_type == "mlp":
        #    model = self.create_model(reg_param, x_train.shape[1])
        #    model.fit(
        #        x_train,
        #        y_train,
        #        sample_weight=train.instance_weights,
        #        epochs=100,
        #        batch_size=100,
        #        verbose=1,
        #    )
        #    return model
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=reg_param, max_iter=1000, verbose=1, solver="liblinear"
            ),
        )
        clf.fit(
            x_train,
            y_train,
            logisticregression__sample_weight=train.instance_weights,
        )
        return clf

    def test_metric(self, test, y_pred):
        test_pred = test.copy()
        if len(y_pred.shape) < 2:
            y_pred = y_pred.reshape(len(y_pred), 1)
        test_pred.labels = y_pred
        metric = ClassificationMetric(
            test,
            test_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        metrics = metric.equal_opportunity_difference()
        print(metric)
        return metrics

    def val_score(self, C_lr):

        scores = []
        metrics = []
        skf = StratifiedKFold(n_splits=5, random_state=2025, shuffle=True)
        for train_index, val_index in skf.split(
            self.train.features, self.train.labels.ravel()
        ):
            train_cv = self.train.subset(train_index)
            test_cv = self.train.subset(val_index)
            test_cv_orig = test_cv.copy()
            if self.drop_prot_feats == True:
                train_cv = self.remove_protected_feats(
                    train_cv, self.privileged_groups[0].keys()
                )
                test_cv = self.remove_protected_feats(
                    test_cv, self.privileged_groups[0].keys()
                )

            # if lr != None:

            #    clf = self.train_model("mlp", train_cv, lr)
            #    score = clf.evaluate(
            #        test_cv.features, test_cv.labels.ravel(), verbose=0
            #    )[1]
            #    labels = self.get_mlp_classes(test_cv, clf)
            #    metric, metrics_bal = self.test_metric(test_cv_orig, labels)

            # if C_svm != None:
            #    clf = self.train_model("svm", train_cv, C_svm)
            #    score = clf.score(test_cv.features, test_cv.labels.ravel())
            #    labels = clf.predict(test_cv.features)
            #    metric = self.test_metric(test_cv_orig, labels)

            clf = self.train_model(train_cv, C_lr)
            score = clf.score(test_cv.features, test_cv.labels.ravel())
            labels = clf.predict(test_cv.features)
            metric = self.test_metric(test_cv_orig, labels)

            scores.append(score)
            metrics.append(metric)

        return np.mean(scores), np.mean(metrics)

    def get_results(self):
        # model_scores_mlp = dict()
        model_scores_lr = dict()
        # mlp_metrics = dict()
        lr_metrics = dict()
        # mlp_metrics_balacc = dict()

        # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2e-2, 3e-2]

        # Function for getting the best hyperparameter according to accuracy
        best_hyperparam_acc = lambda model_scores: max(
            model_scores, key=model_scores.get
        )
        # Function for getting the best hyperparameter with minimum absolute value of
        # fairness score
        best_hyperparam_eqop = lambda model_scores: min(
            model_scores, key=lambda x: abs(model_scores[x])
        )

        # for i in learning_rates:
        #   v(
        #        model_scores_mlp[str(i)],
        #        mlp_metrics[str(i)],
        #        mlp_metrics_balacc[str(i)],
        #    ) = self.val_score(lr=i)

        # mlp_lr_acc = float(best_hyperparam_acc(model_scores_mlp))

        # mlp_lr_eqop = float(best_hyperparam_eqop(mlp_metrics))

        # best_mlp_acc = self.train_model("mlp", self.train, mlp_lr_acc)

        # best_mlp_eqop = self.train_model("mlp", self.train, mlp_lr_eqop)

        for i in [1.0, 100.0, 10000.0, 100000.0, 0.001, 0.0001, 0.00001, 0.000001]:

            model_scores_lr[str(i)], lr_metrics[str(i)] = self.val_score(i)

        lr_C_acc = float(best_hyperparam_acc(model_scores_lr))
        # For learning rate with least parity

        lr_C_eqop = float(best_hyperparam_eqop(lr_metrics))

        best_lr_acc = self.train_model(self.train, lr_C_acc)
        best_lr_eqop = self.train_model(self.train, lr_C_eqop)

        # Testing saved results

        # mlp_dict = {
        #    "learning_rate": list(model_scores_mlp.keys()),
        #    "Average CV Accuracy": list(model_scores_mlp.values()),
        #    "Average Balanced Accuracy": list(mlp_metrics_balacc.values()),
        #    "Equal Opportunity Difference": list(mlp_metrics.values()),
        # }
        lr_dict = {
            "C": list(model_scores_lr.keys()),
            "Average CV Accuracy": list(model_scores_lr.values()),
            "Equal Opportunity Difference": list(lr_metrics.values()),
        }
        model_perf = pd.DataFrame(lr_dict)
        result_df = model_perf
        x_test, y_test = self.test.features, self.test.labels.ravel()
        # mlp_acc_wise = {
        #    "Accuracy": [best_mlp_acc.evaluate(x_test, y_test, verbose=0)[1]],
        #    "Equal Opportunity Difference": [
        #        self.test_metric(
        #            self.test, self.get_mlp_classes(self.test, best_mlp_acc)
        #        )[0]
        #    ],
        #    "Balanced Accuracy": [
        #        self.test_metric(
        #            self.test, self.get_mlp_classes(self.test, best_mlp_acc)
        #        )[1]
        #    ],
        # }

        # mlp_eqop_wise = {
        #    "Accuracy": [best_mlp_eqop.evaluate(x_test, y_test, verbose=0)[1]],
        #    "Equal Opportunity Difference": [
        #        self.test_metric(
        #            self.test, self.get_mlp_classes(self.test, best_mlp_eqop)
        #        )[0]
        #    ],
        #    "Balanced Accuracy": [
        #        self.test_metric(
        #            self.test, self.get_mlp_classes(self.test, best_mlp_eqop)
        #        )[1]
        #    ],
        # }

        lr_eqop_wise = {
            "Accuracy": [best_lr_eqop.score(x_test, y_test)],
            "Equal Opportunity Difference": [
                self.test_metric(self.test, best_lr_eqop.predict(x_test))
            ],
        }

        lr_acc_wise = {
            "Accuracy": [best_lr_acc.score(x_test, y_test)],
            "Equal Opportunity Difference": [
                self.test_metric(self.test, best_lr_acc.predict(x_test))
            ],
        }

        model_perf_test = {
            #    "MLP Accuracy Wise": pd.DataFrame(mlp_acc_wise),
            #    "MLP Fairness Wise": pd.DataFrame(mlp_eqop_wise),
            "LR Accuracy Wise": pd.DataFrame(lr_acc_wise),
            "LR Fairness Wise": pd.DataFrame(lr_eqop_wise),
        }

        test_result = pd.concat(
            model_perf_test.values(), keys=model_perf_test.keys(), axis=1
        )
        if self.reweighing == True:
            result_df.to_pickle(f"Cross_Val_Results_reweighted_{self.suffix}.pkl")
            test_result.to_pickle(f"models_testset_reweighted_{self.suffix}.pkl")
        else:
            result_df.to_pickle(f"Cross_Val_Results_{self.suffix}.pkl")
            test_result.to_pickle(f"models_testset_{self.suffix}.pkl")

        print("Cross Val result: \n")
        print(result_df)
        print("Tets Set Result: \n")
        print(test_result)
        return result_df, test_result
