import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle

from xgboost import XGBRegressor

from scipy.stats import spearmanr

########################################################################################################################


def delete_nan_subjects(data, report):
    nanmat = list()
    for sub in range(0, len(data)):
        if np.any(np.where(np.isnan(data[sub]))):
            nanmat.append(sub)

    for k in reversed(nanmat):
        del data[k]

    report = report.drop(nanmat)

    return data, report


def regression_permutation_test(estimator, cv, X, y, n_perm):

    xgb = XGBRegressor(objective=estimator.objective, learning_rate=estimator.learning_rate, max_depth=estimator.max_depth,
                       min_child_weight=estimator.min_child_weight, n_estimators=estimator.n_estimators,
                       colsample_bytree=estimator.colsample_bytree, subsample=estimator.subsample, gamma=estimator.gamma,
                       tree_method='gpu_hist', random_state=42)

    mae_perm = list()
    for _ in range(0, n_perm):

        y_shuffled = shuffle(y)

        cv_mae = []
        for train_idx, test_idx in cv.split(X):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_shuffled.iloc[train_idx], y_shuffled.iloc[test_idx]

            xgb.fit(X_train, y_train)
            pred_perm = xgb.predict(X_test)

            cv_mae.append(mean_absolute_error(pred_perm, y_test))

        mae_perm.append(np.mean(cv_mae))

    return mae_perm


def regression_feature_selection(X_train, X_test, y_train, idx, model):

    X_train_sel = X_train.iloc[:, idx]
    X_test_sel = X_test.iloc[:, idx]

    model.fit(X_train_sel, y_train)
    pred = model.predict(X_test_sel)

    return pred

