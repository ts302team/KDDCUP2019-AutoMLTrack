'''
Copyright (C) 2019  ts302_team
Jian Sun: yearsj111110@163.com
Chunmeng Zhong: 18801130730@163.com
Hao Zhang: zh_94@outlook.com
Hongyu Jia: jia_hy@outlook.com
Xiao Huang: hx36w35@163.com
Bin Lin: 15951872937@163.com
Zaiyu Pang: pangzaiyu@163.com	

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import datetime

import CONSTANT
import numpy as np
import pandas as pd
from util import *
import collections
from sklearn.model_selection import KFold
from numpy.random import normal

from sklearn.feature_extraction.text import CountVectorizer
from resource_manager import *


class FrequencyCatEncoder(TimeControlObject):
    def __init__(self, params):
        super().__init__(params['all_rows'],params['sample_rows'])
        self.cols = params['cols']
        self.col = self.cols[0]
        self.type = None
        self.name = params['name']

    #@zyptimeit
    def fit_transform(self, X, y=None):
        start_time=time.time()
        curr_feature_map = X[self.col].value_counts()
        X[self.name] = curr_feature_map[X[self.col]].reset_index(drop=True)
        X[self.name] = pd.to_numeric(X[self.name], downcast='unsigned')
        end_time=time.time()
        self.set_col_time_estimate('all', end_time-start_time)


class FrequencyMulcatEncoder(TimeControlObject):
    def __init__(self, params):
        super().__init__(params['all_rows'],params['sample_rows'])
        self.cols = params['cols']
        self.col = self.cols[0]
        self.type = ['sum', 'min', 'max', 'mean', 'count']
        self.name = params['name']

    #@zyptimeit
    def fit_transform(self, X, y=None):
        start_time = time.time()

        tmp = X[self.col].values.tolist()
        cat_count = {}
        for values in tmp:
            for value in values.split(','):
                try:
                    cat_count[value] += 1
                except KeyError:
                    cat_count[value] = 1

        def split_row(x):
            cur_row = [cat_count[v] for v in x.split(",")]
            cur_row_sum = sum(cur_row)
            cur_row_len = len(cur_row)
            cur_row_mean = cur_row_sum/cur_row_len if cur_row_len > 0 else 0
            return cur_row_sum, min(cur_row), max(cur_row), cur_row_mean, cur_row_len #np.std(cur_row), len(cur_row)
        tmp = np.array(list(map(lambda x: split_row(x), tmp)))

        if 'sum' in self.type:
            X[f"{self.name}#sum"] = tmp[:, 0]
            X[f"{self.name}#sum"] = pd.to_numeric(X[f"{self.name}#sum"], downcast='unsigned')
        if 'min' in self.type:
            X[f"{self.name}#min"] = tmp[:, 1]
            X[f"{self.name}#min"] = pd.to_numeric(X[f"{self.name}#min"], downcast='unsigned')
        if 'max' in self.type:
            X[f"{self.name}#max"] = tmp[:, 2]
            X[f"{self.name}#max"] = pd.to_numeric(X[f"{self.name}#max"], downcast='unsigned')
        if 'mean' in self.type:
            X[f"{self.name}#mean"] = tmp[:, 3]
            X[f"{self.name}#mean"] = pd.to_numeric(X[f"{self.name}#mean"], downcast='float')
        if 'count' in self.type:
            X[f"{self.name}#count"] = tmp[:, 4]
            X[f"{self.name}#count"] = pd.to_numeric(X[f"{self.name}#count"], downcast='unsigned')
        del tmp
        end_time = time.time()
        self.set_col_time_estimate('all',end_time-start_time)

class CTREncode(TimeControlObject):
    def __init__(self, params, add_random=False, rmean=0, rstd=0.1, alpha=0, folds=5, sample_rate=0.5):
        super().__init__(params['all_rows'],params['sample_rows'])
        self.type = params['type']
        self.name = params['name']
        self.add_random = add_random
        self.rmean = rmean
        self.rstd = rstd
        self.folds = folds
        self.alpha = alpha
        self.ctr_map = {}
        self.sample_rate = sample_rate
        self.target_col = 'label'
        self.target_mean_global = 0
        self.col = None
        self.sign = True

    def kfold_process(self, data_tuple):
        df_for_estimation = data_tuple[0]
        df_estimated = data_tuple[1]
        '''
        click_rate = df_for_estimation.groupby(self.col)[self.target_col].mean()
        alpha, beta = self.getBayesSmoothParam(click_rate)
        # getting means on data for estimation (all folds except estimated)
        click = df_for_estimation.groupby(self.col)[self.target_col].sum()
        exposure = df_for_estimation.groupby(self.col)[self.target_col].size()
        '''
        tmp = df_for_estimation[[self.col, self.target_col]].groupby(self.col)
        click_rate = tmp[self.target_col].mean()
        alpha, beta = self.getBayesSmoothParam(click_rate)
        # getting means on data for estimation (all folds except estimated)
        click = tmp[self.target_col].sum()
        exposure = tmp[self.target_col].size()
        del tmp

        if alpha == 0:
            target_means_cats_adj = click / exposure
        else:
            target_means_cats_adj = (click + alpha) / (exposure + alpha + beta)

        # Mapping means to estimated fold
        encoded_col_train_part = df_estimated[self.col].map(target_means_cats_adj)
        if self.add_random:
            encoded_col_train_part = encoded_col_train_part + normal(loc=self.rmean, scale=self.rstd,
                                                                     size=(encoded_col_train_part.shape[0]))
        return encoded_col_train_part

    def data_sample(self, X: pd.DataFrame, rate: float = 0.5):
        # -> (pd.DataFrame, pd.Series):
        nrows = int(len(X) * rate)
        return X.sample(nrows, random_state=1) if len(X.index) > nrows else X

    def getBayesSmoothParam(self, origion_rate):
        origion_rate_mean = origion_rate.mean()
        origion_rate_var = origion_rate.var()
        if origion_rate_mean == 0 or origion_rate_var == 0 or np.isnan(origion_rate_var):
            return origion_rate_mean, 0

        alpha = origion_rate_mean / origion_rate_var * (origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)
        beta = (1 - origion_rate_mean) / origion_rate_var * (
                origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)

        return alpha, beta

    def fit_transform(self, X, y):
        if self.type is None or len(self.type) == 0:
            return
        timer = Timer()
        if len(X) != len(y):

            ret_train = self.helper(X.iloc[0:CONSTANT.TRAIN_LEN], y)
            ret_test = self.transform(X.iloc[CONSTANT.TRAIN_LEN:], y)
            ret = pd.concat([ret_train, ret_test], axis=0, ignore_index=True)
            X[ret.columns] = ret
        else:
            ret = self.helper(X, y)
            X[ret.columns] = ret

    #@zyptimeit
    def helper(self, X, y):
        ret = pd.DataFrame()

        start_time=time.time()
        self.target_mean_global = y.mean()
        if self.type is None or len(self.type) == 0:
            end_time = time.time()
            self.set_col_time_estimate('extra', end_time - start_time)
            return

        self.target_col = 'label'

        X_cat = X[self.type]
        X_cat[self.target_col] = y

        kfold = KFold(n_splits=self.folds, random_state=None, shuffle=False)

        def split_data(myindex):
            tr_in = myindex[0]
            val_ind = myindex[1]
            df_for_estimation, df_estimated = X_cat.iloc[tr_in], X_cat.iloc[val_ind]
            df_for_estimation = self.data_sample(df_for_estimation, self.sample_rate)
            return df_for_estimation, df_estimated

        data_tuples = list(map(split_data, kfold.split(X))) #问题：如果数据量很大，会爆内存

        end_time=time.time()
        self.set_col_time_estimate('extra',end_time-start_time)

        for col in self.type:
            start_time=time.time()
            self.col = col

            # -------------compute ctr codes for prediction-----------------------
            tmp = X_cat.groupby(col)
            click_rate = tmp[self.target_col].mean()
            alpha, beta = self.getBayesSmoothParam(click_rate)
            # getting means on data for estimation (all folds except estimated)
            click = tmp[self.target_col].sum()
            exposure = tmp[self.target_col].size()
            del tmp

            if alpha == 0:
                target_means_cats_adj = click / exposure
            else:
                target_means_cats_adj = (click + alpha) / (exposure + alpha + beta)

            # Mapping means to test data
            self.ctr_map[col] = target_means_cats_adj
            #timer.check('ctr_map')
            # ------------------------------------------------------

            # ----- compute train dataset ctr codes---------------
            small_data_tuples = list(map(lambda data_tuple: tuple(map(lambda data: data[[col, self.target_col]], data_tuple)), data_tuples))

            # ----non parallel version-----
            parts = list(map(self.kfold_process, small_data_tuples))

            encoded_col_train = pd.concat(parts, axis=0)
            encoded_col_train.fillna(self.target_mean_global, inplace=True)
            ret[f'{self.name}#{col}'] = encoded_col_train
            ret[f"{self.name}#{col}"] = pd.to_numeric(ret[f"{self.name}#{col}"], downcast='float')
            # ---------------------------------------------------
            end_time = time.time()
            self.set_col_time_estimate(col,end_time-start_time)

        #X.drop(self.target_col, axis=1, inplace=True)
        del data_tuples
        return ret


    def transform(self, X, y=None):
        ret = pd.DataFrame()
        for col in self.type:
            encoded_col_test = self.ctr_map[col].reindex(X[col])
            encoded_col_test = encoded_col_test.fillna(self.target_mean_global).reset_index(drop=True)
            ret[f'{self.name}#{col}'] = encoded_col_test
            ret[f"{self.name}#{col}"] = pd.to_numeric(ret[f"{self.name}#{col}"], downcast='float')
        return ret

class DatatimeEncoder(TimeControlObject):
    def __init__(self, params):
        super().__init__(params['all_rows'],params['sample_rows'])
        self.name = params['name']
        self.cols = params['cols']
        self.col = self.cols[0]
        self.type = None

    #@zyptimeit
    def fit_transform(self, X, y=None):
        """
        zypang
        :param X:
        :param y:
        :return:
        """
        start_time=time.time()
        #zyp: tmp = pd.to_datetime(X[self.col]).astype('category')
        tmp = X[self.col]
        #print(tmp.dtype)
        if self.type is None:
            self.type = ['year', 'month', 'day', 'hour', 'minute', 'second', 'millisecond', 'microsecond']
        type_dict = {
            'year':'uint16',
            'month':'uint8',
            'day':'uint8',
            'hour':'uint8',
            'minute':'uint8',
            'second':'uint8',
            'millisecond':'uint16',
            'microsecond': 'uint16',
        }
        for item in self.type:
            if item == 'millisecond':
                X[f"{self.name}#{item}"] = getattr(tmp.dt, 'microsecond').astype(type_dict[item]) // 1000
            else:
                X[f"{self.name}#{item}"] = getattr(tmp.dt, item).astype(type_dict[item])

        del tmp
        end_time=time.time()
        self.set_col_time_estimate('all',end_time-start_time)

class BinaryEncoder(TimeControlObject):
    def __init__(self, params):
        super().__init__(params['all_rows'],params['sample_rows'])
        self.name = params['name']
        self.cols = params['cols']
        self.col = self.cols[0]
        self.type = None

    #@zyptimeit
    def fit_transform(self, X, y=None):
        def avg(x):
            avg = 0
            for v in x:
                if v == '':
                    continue
                avg += int(v)
            return avg / len(x)

        start_time=time.time()
        if self.type is None:
            self.type = ['ave']
        X[f"{self.name}#ave"] = X[self.col].apply(lambda x: avg([int(v) for v in x.split(',')])) #.astype('float16')
        X[f"{self.name}#ave"] = pd.to_numeric(X[f"{self.name}#ave"], downcast='float')
        end_time=time.time()
        self.set_col_time_estimate('all',end_time-start_time)

class SecondOrderCountEncoder(TimeControlObject):
    def __init__(self, params):
        super().__init__(params['all_rows'],params['sample_rows'])
        self.name = params['name']
        self.cols = params['cols']
        self.key = self.cols[0]
        self.other = self.cols[1]
        self.key_count = None
        self.type = ['count']

    #@zyptimeit
    def fit_transform(self, X, y=None):
        start_time=0
        if 'count' in self.type:
            start_time = time.time()
            tmp_series = X[self.key] * CONSTANT.MAX_CAT_NUM + X[self.other]
            self.count_map = tmp_series.value_counts(sort=False)
            X[f'{self.name}#count'] = self.count_map[tmp_series].reset_index(drop=True)
            X[f'{self.name}#count']= pd.to_numeric(X[f'{self.name}#count'], downcast='unsigned')
            end_time=time.time()
            self.set_col_time_estimate('count', end_time-start_time)

class SecondOrderNuniqueEncoder(TimeControlObject):
    def __init__(self, params):
        super().__init__(params['all_rows'], params['sample_rows'])
        self.name = params['name']
        self.key = params['key']
        self.type = params['type']

    #@zyptimeit
    def fit_transform(self, X, y=None):
        start_time=time.time()
        if len(self.type) == 1 and self.type[0] == self.key:
            return

        if self.key not in self.type:
            self.type.append(self.key)

        feature_map = X[self.type].groupby(self.key).nunique()
        end_time=time.time()
        self.set_col_time_estimate('extra', end_time-start_time)
        for other in self.type:
            if other == self.key:
                self.set_col_time_estimate(other, 0)
                continue
            start_time=time.time()
            X[f'{self.name}#{other}'] = feature_map[other][X[self.key]].reset_index(drop=True)
            X[f'{self.name}#{other}'] = pd.to_numeric(X[f'{self.name}#{other}'], downcast='unsigned')
            end_time=time.time()
            self.set_col_time_estimate(other, end_time-start_time)

class SecondOrderCatCatEncoder(TimeControlObject):
    def __init__(self, params):
        super().__init__(params['all_rows'],params['sample_rows'])
        self.name = params['name']
        self.key = params['key']
        self.other = params['other']
        self.type = []

    #@zyptimeit
    def fit_transform(self, X, y=None):
        start_time=time.time()
        X[self.name] = X[self.key]*CONSTANT.MAX_CAT_NUM + X[self.other]
        end_time = time.time()
        self.set_col_time_estimate("all", end_time - start_time)
