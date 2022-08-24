import numpy as np
import pandas as pd
import time
from sklearn.metrics import average_precision_score

try:
    from utils import *
except ImportError:
    from .utils import *


def disparate_impact(y1, y2):
    return sum(y1) / len(y1) / (sum(y2) / len(y2))


class Linear:

    def __init__(self, objective):
        """
        Train the model via linear regression
    
        Parameters
        ----------
        objective: Object

        """

        try:
            train_data = objective.train_data
            test_data = objective.test_data
            protected = objective.protected
            percentiles_list = objective.percentiles_list
            percentiles_str_list = objective.percentiles_str_list
            threshold_test = objective.threshold_test
        except ValueError:
            raise ValueError('Failed to load the objective')

        start_time = time.time()
        not_train_vars = get_not_train_vars()
        from sklearn.linear_model import LinearRegression
        model_ols = LinearRegression(n_jobs=-1)
        model_ols.fit(train_data.drop(columns=not_train_vars),
                      train_data['TOTCHG'])
        totchg_pred = \
            model_ols.predict(test_data.drop(columns=not_train_vars))
        threshold_pred = list()
        statistics_test = \
            pd.Series(totchg_pred).describe(percentiles=percentiles_list)
        for _ in percentiles_str_list:
            threshold_pred.append(statistics_test[_])
        AUPRC = dict()
        DI = dict()
        DI_list = dict()

        for i in range(len(percentiles_str_list)):
            y_test = test_data['TOTCHG']\
                    .gt(threshold_test[i]).astype('int')
            y_pred = totchg_pred > threshold_pred[i].astype('int')
            AUPRC[percentiles_str_list[i]] = \
                round(average_precision_score(y_test, y_pred,
                      average='macro'), 6)
            if protected == 'RACE':
                range_protected = range(1, 7)
            if protected == 'ZIPINC_QRTL':
                range_protected = range(1, 5)
            DI_list = list()
            for j in range_protected:

                test_1 = test_data[test_data[protected] == j]
                test_2 = test_data[test_data[protected] != j]
                totchg_pred_1 = \
                    model_ols.predict(test_1.drop(columns=not_train_vars))
                totchg_pred_2 = \
                    model_ols.predict(test_2.drop(columns=not_train_vars))
                y_pred_1 = totchg_pred_1 \
                    > threshold_pred[i].astype('int')
                y_pred_2 = totchg_pred_2 \
                    > threshold_pred[i].astype('int')
                DI_list.append(disparate_impact(y_pred_1, y_pred_2))
            DI[percentiles_str_list[i]] = \
                round(np.exp(-np.mean(np.absolute(np.log(DI_list)))), 6)
            DI[percentiles_str_list[i]+'_list'] = \
                [round(_, 6) for _ in DI_list]
        self.AUPRC = AUPRC
        self.DI = DI
        self.time = time.time() - start_time


class Lasso:

    def __init__(self, objective, seed=None):
        """
        Train the model via Lasso regression
    
        Parameters
        ----------
        objective: Object

        """

        try:
            train_data = objective.train_data
            test_data = objective.test_data
            protected = objective.protected
            percentiles_list = objective.percentiles_list
            percentiles_str_list = objective.percentiles_str_list
            threshold_test = objective.threshold_test
        except ValueError:
            raise ValueError('Failed to load the objective')

        start_time = time.time()
        not_train_vars = get_not_train_vars()
        from sklearn.linear_model import Lasso

        model_lasso = Lasso(max_iter=1e4, tol=1e-3, random_state=seed,
                            selection='random')
        model_lasso.fit(train_data.drop(columns=not_train_vars),
                        train_data['TOTCHG'])
        totchg_pred = \
            model_lasso.predict(test_data.drop(columns=not_train_vars))
        threshold_pred = list()
        statistics_test = \
            pd.Series(totchg_pred).describe(percentiles=percentiles_list)
        for _ in percentiles_str_list:
            threshold_pred.append(statistics_test[_])
        AUPRC = dict()
        DI = dict()

        for i in range(len(percentiles_str_list)):
            y_test = test_data['TOTCHG']\
                    .gt(threshold_test[i]).astype('int')
            y_pred = totchg_pred > threshold_pred[i].astype('int')
            AUPRC[percentiles_str_list[i]] = \
                round(average_precision_score(y_test, y_pred,
                      average='macro'), 6)
            if protected == 'RACE':
                range_protected = range(1, 7)
            if protected == 'ZIPINC_QRTL':
                range_protected = range(1, 5)
            DI_list = list()
            for j in range_protected:
                test_1 = test_data[test_data[protected] == j]
                test_2 = test_data[test_data[protected] != j]
                totchg_pred_1 = \
                    model_lasso.predict(test_1.drop(columns=not_train_vars))
                totchg_pred_2 = \
                    model_lasso.predict(test_2.drop(columns=not_train_vars))
                y_pred_1 = totchg_pred_1 \
                    > threshold_pred[i].astype('int')
                y_pred_2 = totchg_pred_2 \
                    > threshold_pred[i].astype('int')
                DI_list.append(disparate_impact(y_pred_1, y_pred_2))
            DI[percentiles_str_list[i]] = \
                round(np.exp(-np.mean(np.absolute(np.log(DI_list)))), 6)
            DI[percentiles_str_list[i]+'_list'] = \
                [round(_, 6) for _ in DI_list]
        self.AUPRC = AUPRC
        self.DI = DI
        self.time = time.time() - start_time


class ElasticNet:

    def __init__(self, objective, seed=None):
        """
        Train the model via Elastic Net regression
    
        Parameters
        ----------
        objective: Object

        """

        try:
            train_data = objective.train_data
            test_data = objective.test_data
            protected = objective.protected
            percentiles_list = objective.percentiles_list
            percentiles_str_list = objective.percentiles_str_list
            threshold_test = objective.threshold_test
        except ValueError:
            raise ValueError('Failed to load the objective')

        start_time = time.time()
        not_train_vars = get_not_train_vars()
        from sklearn.linear_model import ElasticNet

        model_elastic = ElasticNet(max_iter=1e4, tol=1e-3,
                                   random_state=seed, selection='random'
                                   )
        model_elastic.fit(train_data.drop(columns=not_train_vars),
                          train_data['TOTCHG'])
        totchg_pred = \
            model_elastic.predict(test_data.drop(columns=not_train_vars))
        threshold_pred = list()
        statistics_test = \
            pd.Series(totchg_pred).describe(percentiles=percentiles_list)
        for _ in percentiles_str_list:
            threshold_pred.append(statistics_test[_])
        AUPRC = dict()
        DI = dict()

        for i in range(len(percentiles_str_list)):
            y_test = test_data['TOTCHG']\
                    .gt(threshold_test[i]).astype('int')
            y_pred = totchg_pred > threshold_pred[i].astype('int')
            AUPRC[percentiles_str_list[i]] = \
                round(average_precision_score(y_test, y_pred,
                      average='macro'), 6)
            if protected == 'RACE':
                range_protected = range(1, 7)
            if protected == 'ZIPINC_QRTL':
                range_protected = range(1, 5)
            DI_list = list()
            for j in range_protected:
                test_1 = test_data[test_data[protected] == j]
                test_2 = test_data[test_data[protected] != j]
                totchg_pred_1 = \
                    model_elastic.predict(test_1.drop(columns=not_train_vars))
                totchg_pred_2 = \
                    model_elastic.predict(test_2.drop(columns=not_train_vars))
                y_pred_1 = totchg_pred_1 \
                    > threshold_pred[i].astype('int')
                y_pred_2 = totchg_pred_2 \
                    > threshold_pred[i].astype('int')
                DI_list.append(disparate_impact(y_pred_1, y_pred_2))
            DI[percentiles_str_list[i]] = \
                round(np.exp(-np.mean(np.absolute(np.log(DI_list)))), 6)
            DI[percentiles_str_list[i]+'_list'] = \
                [round(_, 6) for _ in DI_list]
        self.AUPRC = AUPRC
        self.DI = DI
        self.time = time.time() - start_time


class Xgboost:

    def __init__(self, objective, seed=None):
        """
        Train the model via extreme gradient boosting
    
        Parameters
        ----------
        objective: Object

        """

        try:
            train_data = objective.train_data
            test_data = objective.test_data
            protected = objective.protected
            percentiles_list = objective.percentiles_list
            percentiles_str_list = objective.percentiles_str_list
            threshold_test = objective.threshold_test
        except ValueError:
            raise ValueError('Failed to load the objective')

        start_time = time.time()
        not_train_vars = get_not_train_vars()
        import xgboost as xgb
        model_xgb = xgb.XGBRegressor(eta=1e-1, max_depth=8,
                                     random_state=seed,
                                     objective='reg:squarederror')
        model_xgb.fit(train_data.drop(columns=not_train_vars),
                      train_data['TOTCHG'])
        totchg_pred = \
            model_xgb.predict(test_data.drop(columns=not_train_vars))
        threshold_pred = list()
        statistics_test = \
            pd.Series(totchg_pred).describe(percentiles=percentiles_list)
        for _ in percentiles_str_list:
            threshold_pred.append(statistics_test[_])
        AUPRC = dict()
        DI = dict()

        for i in range(len(percentiles_str_list)):
            y_test = test_data['TOTCHG']\
                    .gt(threshold_test[i]).astype('int')
            y_pred = totchg_pred > threshold_pred[i].astype('int')
            AUPRC[percentiles_str_list[i]] = \
                round(average_precision_score(y_test, y_pred,
                      average='macro'), 6)
            if protected == 'RACE':
                range_protected = range(1, 7)
            if protected == 'ZIPINC_QRTL':
                range_protected = range(1, 5)
            DI_list = list()
            for j in range_protected:
                test_1 = test_data[test_data[protected] == j]
                test_2 = test_data[test_data[protected] != j]
                totchg_pred_1 = \
                    model_xgb.predict(test_1.drop(columns=not_train_vars))
                totchg_pred_2 = \
                    model_xgb.predict(test_2.drop(columns=not_train_vars))
                y_pred_1 = totchg_pred_1 \
                    > threshold_pred[i].astype('int')
                y_pred_2 = totchg_pred_2 \
                    > threshold_pred[i].astype('int')
                DI_list.append(disparate_impact(y_pred_1, y_pred_2))
            DI[percentiles_str_list[i]] = \
                round(np.exp(-np.mean(np.absolute(np.log(DI_list)))), 6)
            DI[percentiles_str_list[i]+'_list'] = \
                [round(_, 6) for _ in DI_list]
        self.AUPRC = AUPRC
        self.DI = DI
        self.time = time.time() - start_time


class NN:

    def __init__(self, objective, seed=None):
        """
        Train the model via artificial neural network
    
        Parameters
        ----------
        objective: Object

        """

        try:
            train_data = objective.train_data
            test_data = objective.test_data
            protected = objective.protected
            percentiles_list = objective.percentiles_list
            percentiles_str_list = objective.percentiles_str_list
            threshold_test = objective.threshold_test
        except ValueError:
            raise ValueError('Failed to load the objective')

        start_time = time.time()
        not_train_vars = get_not_train_vars()
        from sklearn.neural_network import MLPRegressor
        model_nn = MLPRegressor(hidden_layer_sizes=(50, 2), 
                                learning_rate_init=1e-1, 
                                learning_rate='adaptive',
                                n_iter_no_change=5,
                                random_state=seed)
        model_nn.fit(train_data.drop(columns=not_train_vars),
                     train_data['TOTCHG'])
        totchg_pred = \
            model_nn.predict(test_data.drop(columns=not_train_vars))
        threshold_pred = list()
        statistics_test = \
            pd.Series(totchg_pred).describe(percentiles=percentiles_list)
        for _ in percentiles_str_list:
            threshold_pred.append(statistics_test[_])
        AUPRC = dict()
        DI = dict()

        for i in range(len(percentiles_str_list)):
            y_test = test_data['TOTCHG']\
                    .gt(threshold_test[i]).astype('int')
            y_pred = totchg_pred > threshold_pred[i].astype('int')
            AUPRC[percentiles_str_list[i]] = \
                round(average_precision_score(y_test, y_pred,
                      average='macro'), 6)
            if protected == 'RACE':
                range_protected = range(1, 7)
            if protected == 'ZIPINC_QRTL':
                range_protected = range(1, 5)
            DI_list = list()
            for j in range_protected:
                test_1 = test_data[test_data[protected] == j]
                test_2 = test_data[test_data[protected] != j]
                totchg_pred_1 = \
                    model_nn.predict(test_1.drop(columns=not_train_vars))
                totchg_pred_2 = \
                    model_nn.predict(test_2.drop(columns=not_train_vars))
                y_pred_1 = totchg_pred_1 \
                    > threshold_pred[i].astype('int')
                y_pred_2 = totchg_pred_2 \
                    > threshold_pred[i].astype('int')
                DI_list.append(disparate_impact(y_pred_1, y_pred_2))
            DI[percentiles_str_list[i]] = \
                round(np.exp(-np.mean(np.absolute(np.log(DI_list)))), 6)
            DI[percentiles_str_list[i]+'_list'] = \
                [round(_, 6) for _ in DI_list]
        self.AUPRC = AUPRC
        self.DI = DI
        self.time = time.time() - start_time


class Logistic:

    def __init__(self, objective):
        """
        Train the model via logistic regression
    
        Parameters
        ----------
        objective: Object

        """

        try:
            train_data = objective.train_data
            test_data = objective.test_data
            protected = objective.protected
            percentiles_str_list = objective.percentiles_str_list
            threshold_train = objective.threshold_train
            threshold_test = objective.threshold_test
        except ValueError:
            raise ValueError('Failed to load the objective')

        start_time = time.time()
        not_train_vars = get_not_train_vars()
        from sklearn.linear_model import LogisticRegression
        model_rl = LogisticRegression(max_iter=1e4)
        AUPRC = dict()
        DI = dict()

        for i in range(len(percentiles_str_list)):
            AUPRC_list = list()
            y_train = train_data['TOTCHG']\
                .gt(threshold_train[i]).astype('int')
            y_test = test_data['TOTCHG']\
                .gt(threshold_test[i]).astype('int')
            model_rl.fit(train_data.drop(columns=not_train_vars),
                         y_train)
            y_pred = \
                model_rl.predict(test_data.drop(columns=not_train_vars))
            AUPRC_list.append(round(average_precision_score(y_test,
                              y_pred, average='macro'), 6))
            DI_list = list()
            if protected == 'RACE':
                range_protected = range(1, 7)
            if protected == 'ZIPINC_QRTL':
                range_protected = range(1, 5)

            for j in range_protected:
                test_1 = test_data[test_data[protected] == j]
                test_2 = test_data[test_data[protected] != j]
                y_pred_1 = \
                    model_rl.predict(test_1.drop(columns=not_train_vars))
                y_pred_2 = \
                    model_rl.predict(test_2.drop(columns=not_train_vars))
                DI_list.append(disparate_impact(y_pred_1, y_pred_2))
            DI[percentiles_str_list[i]] = \
                round(np.exp(-np.mean(np.absolute(np.log(DI_list)))), 6)
            DI[percentiles_str_list[i]+'_list'] = \
                [round(_, 6) for _ in DI_list]
            AUPRC[percentiles_str_list[i]] = np.mean(AUPRC_list)
        self.AUPRC = AUPRC
        self.DI = DI
        self.time = time.time() - start_time


class SVM:

    def __init__(self, objective, seed=None):
        """
        Train the model via support vector machine
    
        Parameters
        ----------
        objective: Object

        """

        try:
            train_data = objective.train_data
            test_data = objective.test_data
            protected = objective.protected
            percentiles_str_list = objective.percentiles_str_list
            threshold_train = objective.threshold_train
            threshold_test = objective.threshold_test
        except ValueError:
            raise ValueError('Failed to load the objective')

        start_time = time.time()
        not_train_vars = get_not_train_vars()
        from sklearn.linear_model import SGDClassifier
        model_svm = SGDClassifier(
            loss='hinge',
            alpha=1e-1,
            max_iter=1e3,
            n_jobs=-1,
            random_state=seed,
            class_weight='balanced',
            )
        AUPRC = dict()
        DI = dict()
        
        for i in range(len(percentiles_str_list)):
            AUPRC_list = list()
            y_train = train_data['TOTCHG']\
                .gt(threshold_train[i]).astype('int')
            y_test = test_data['TOTCHG']\
                .gt(threshold_test[i]).astype('int')
            model_svm.fit(train_data.drop(columns=not_train_vars),
                          y_train)
            y_pred = \
                model_svm.predict(test_data.drop(columns=not_train_vars))
            AUPRC_list.append(round(average_precision_score(y_test,
                              y_pred, average='macro'), 6))
            DI_list = list()
            if protected == 'RACE':
                range_protected = range(1, 7)
            if protected == 'ZIPINC_QRTL':
                range_protected = range(1, 5)
            for j in range_protected:
                test_1 = test_data[test_data[protected] == j]
                test_2 = test_data[test_data[protected] != j]
                y_pred_1 = \
                    model_svm.predict(test_1.drop(columns=not_train_vars))
                y_pred_2 = \
                    model_svm.predict(test_2.drop(columns=not_train_vars))
                DI_list.append(disparate_impact(y_pred_1, y_pred_2))
            DI[percentiles_str_list[i]] = \
                round(np.exp(-np.mean(np.absolute(np.log(DI_list)))), 6)
            DI[percentiles_str_list[i]+'_list'] = \
                [round(_, 6) for _ in DI_list]
            AUPRC[percentiles_str_list[i]] = np.mean(AUPRC_list)
        self.AUPRC = AUPRC
        self.DI = DI
        self.time = time.time() - start_time
