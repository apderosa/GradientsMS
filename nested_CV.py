from time import time
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pickle

from utils import regression_permutation_test, regression_feature_selection


def regression_nested_CV(dataset, features):

    start = time()

    # Define hyperparameters space
    space = dict()
    space['learning_rate'] = [0.01, 0.05, 0.1]
    space['max_depth'] = [3, 5, 7]
    space['min_child_weight'] = [1, 3, 5]
    space['n_estimators'] = [20, 60, 100]
    space['colsample_bytree'] = [0.6, 0.8, 1]
    space['subsample'] = [0.6, 0.8, 1]

    X = dataset.drop('SDMT', axis=1)
    y = dataset[['SDMT']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    val_mae = []
    test_mae = []
    perm = []
    importances = []
    y_pred_val = []
    y_true = []
    y_pred = []
    perm_mae = []
    for rep in range(0, 10):

        print('\n\033[1m' + '### Repetition number {} ###'.format(rep + 1) + '\033[0m')

        # Dataset splitting
        outer_cv = KFold(n_splits=5, random_state=rep, shuffle=True)
        inner_cv = KFold(n_splits=5, random_state=rep, shuffle=True)

        outer_mae = np.full((outer_cv.n_splits, X.shape[1]), np.nan)
        outer_pred = [[] for _ in range(0, 5)]
        y_val_predict = np.empty((len(y_train), 1))
        models = []
        outer_iter = 1
        for outer_train_idx, outer_test_idx in outer_cv.split(X_train):

            print(' ')
            print('Outfold {}'.format(outer_iter))

            X_train_outer, X_test_outer = X_train.iloc[outer_train_idx], X_train.iloc[outer_test_idx]
            y_train_outer, y_test_outer = y_train.iloc[outer_train_idx], y_train.iloc[outer_test_idx]

            # Inner fold - Hyperparameters tuning
            xgb = XGBRegressor(tree_method='gpu_hist', random_state=42)
            xgb_rs = GridSearchCV(xgb, space, scoring='neg_mean_absolute_error', n_jobs=100,
                                  refit=True, cv=inner_cv, verbose=1)
            result = xgb_rs.fit(X_train_outer, y_train_outer)

            # Select best inner model
            best_model_inner = result.best_estimator_
            models.append(best_model_inner)

            best_model_inner.fit(X_train_outer, y_train_outer)
            feature_importances = best_model_inner.feature_importances_
            importances.append(feature_importances)

            sorted_idx = np.argsort(feature_importances)[::-1]

            # Outer loop - Features selection & model validation
            top_feat = np.array([])
            feat = 0
            for nidx in sorted_idx:
                top_feat = np.append(top_feat, nidx)
                sel_pred = regression_feature_selection(X_train_outer, X_test_outer, y_train_outer, top_feat, best_model_inner)

                outer_pred[outer_iter-1].append(sel_pred)
                outer_mae[outer_iter-1, feat] = mean_absolute_error(y_test_outer, sel_pred)
                feat += 1

            outer_iter += 1

        mean_outer_mae = np.mean(outer_mae, axis=0)
        idx = np.argmin(mean_outer_mae)

        val_mae.append(mean_outer_mae[idx])

        outer_idx = 0
        for _, outer_test_idx in outer_cv.split(X_train):

            y_val_predict[outer_test_idx] = outer_pred[outer_idx][idx].reshape(-1, 1)
            outer_idx += 1

        y_pred_val.append(y_val_predict)

        # Select best outer model
        idx_model = np.argmin(outer_mae[:, idx])
        best_model_outer = models[idx_model]

        best_model_outer.fit(X_train.iloc[:, sorted_idx[:idx+1]], y_train)
        y_hat = best_model_outer.predict(X_test.iloc[:, sorted_idx[:idx+1]])

        test_mae.append(mean_absolute_error(y_test, y_hat))

        # Permutation testing
        perm_score = regression_permutation_test(best_model_outer, outer_cv, X_train.iloc[:, sorted_idx[:idx+1]],
                                                 y_train, 1000)
        perm_mae.append(perm_score)
        p_perm = (np.sum(perm_score <= val_mae[rep]) + 1) / 1001
        perm.append(p_perm)

        y_true.append(y_test)
        y_pred.append(y_hat)

    end = time()
    print('\n Done!')
    print('Time elapsed: ', end - start)

    with open(features + '_regression.pkl', 'wb') as f:
        pickle.dump([importances, y_pred_val, val_mae, test_mae, perm, y_true, y_pred, perm_mae], f)
