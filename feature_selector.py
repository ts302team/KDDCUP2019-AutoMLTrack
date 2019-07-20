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

import lightgbm as lgb
import copy
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from typing import Dict
from util import *

import CONSTANT


class FeatureSelector:
    """
    参数：
        data : dataframe 训练数据
        labels : array or series, 标签数据，可为None
    """

    def __init__(self, data, labels=None):
        self.labels = labels
        # self.data = copy.deepcopy(data)
        self.data = data
        self.base_features = list(data.columns)
        print(f'FeatureSelector base features:{len(self.base_features)}')

        self.feature_importances = None

        self.ops = {}

    @timeit
    def identify_collinear(self, correlation_threshold):
        """基于皮尔逊相关系数识别共线特征"""
        num_col = [c for c in self.data if c.startswith(CONSTANT.NUMERICAL_PREFIX) or c.startswith(CONSTANT.CAT_INT_PREFIX)]  #只对数值类型做共线特征
        if len(num_col) == 0:
            self.ops['collinear'] = []
            return

        if len(num_col) > 1000 and len(self.data) > 20000:
            data = data_sample(self.data, 8000)
        elif len(self.data) >= 50000:
            data = data_sample(self.data, 10000)
        else:
            data = self.data

        corr_matrix = data[num_col].corr()

        # 获取相关性矩阵上三角
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column].abs() >= correlation_threshold)]
        self.ops['collinear'] = to_drop
        print('%d features with a correlation magnitude greater than %0.3f.' % (len(self.ops['collinear']), correlation_threshold))

    @timeit
    def hyperopt_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                          params: Dict):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "scale_pos_weight": hp.uniform('x', 0, 5),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 500,
                              valid_data, early_stopping_rounds=100, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            feature_importance_df = pd.DataFrame()
            feature_importance_df["features"] = X_train.columns
            feature_importance_df["importance_gain"] = model.feature_importance(importance_type='gain')
            record_zero_importance = feature_importance_df[feature_importance_df["importance_gain"] == 0.0]
            to_drop = list(record_zero_importance['features'])

            # in classification, less is better
            return {'loss': -score, 'status': STATUS_OK, "drop_feature": to_drop, "best_iter": model.best_iteration}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=10, verbose=1,
                             rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"hyperopt auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        drop_feature = set(X_train.columns.tolist())
        for result in trials.results:
            drop_feature = drop_feature & set(result['drop_feature'])
        return drop_feature, hyperparams

    @timeit
    def identify_zero_importance(self, valid_ratio=0.2, not_need_list = []):
        """
        识别模型训练中零重要性特征
        目前使用automl中训练方式，用少量数据集和迭代次数识别零重要性特征
        """
        num_feats = [col for col in self.data.columns if
                               not (col.startswith(CONSTANT.CATEGORY_PREFIX) or col.startswith(
                                   CONSTANT.MULTI_CAT_PREFIX) or col.startswith(CONSTANT.TIME_PREFIX)) and col not in not_need_list]
        if len(num_feats) == 0:
            self.ops['zero_importance'] = []
            return None

        X_train, X_val, y_train, y_val = data_split(self.data[num_feats], self.labels, valid_ratio)

        print(f'x_train_len: {len(X_train)} x_val_len: {len(X_val)}')
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4,
            "max_depth": 6,
            "num_leaves": 32,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }

        drop_feature, hyperparams = self.hyperopt_lightgbm(X_train, y_train, X_val, y_val, params)
        #for c in drop_feature:
        #    print(c)
        self.ops['zero_importance'] = list(drop_feature)

        print('%d features with zero importance.\n' % len(self.ops['zero_importance']))
        return {**params, **hyperparams}

    @timeit
    def identify_low_importance(self, top_k=None, top_ratio=None, free_list=[], valid_ratio=0.2, num_boost_round=500, params=None, not_need_list=[]):
        log(f"free_list length : {len(free_list)}")
        log(f"top_k : {top_k}")
        """
        根据比例 or 个数选取重要性靠前特征（识别出重要性靠后特征）
        top_k:      个数选取
        top_ratio:  比例选取
        均不为None则选取两者中较少特征
        """
        num_feats = [col for col in self.data.columns if
                               not (col.startswith(CONSTANT.CATEGORY_PREFIX) or col.startswith(CONSTANT.MULTI_CAT_PREFIX) \
                                    or col.startswith(CONSTANT.TIME_PREFIX) or col.startswith(CONSTANT.CATEGORY_HIGH_PREFIX)) and col not in not_need_list]
        if len(num_feats) == 0:
            self.ops['low_importance'] = []
            self.ops['needed_cols'] = free_list
            return

        X_train, X_val, y_train, y_val = data_split(self.data[num_feats], self.labels, valid_ratio)
        if params == None:
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "seed": 1,
                "num_threads": 4,
                "max_depth": 6,
                "num_leaves": 32,
                "feature_fraction": 0.6,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "learning_rate": 0.01,
                #"min_child_weight": 5,
            }
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(params,
                          train_data,
                          num_boost_round,
                          valid_data,
                          verbose_eval=0)

        self.feature_importances = pd.DataFrame()
        self.feature_importances['features'] = X_train.columns
        self.feature_importances['importance_gain'] = model.feature_importance(importance_type='gain')
        self.feature_importances.sort_values('importance_gain', inplace=True, ascending=False)

        total_features_without_zero = len(self.feature_importances[self.feature_importances['importance_gain'] > 0])
        needed_cols = free_list.copy()
        for i in range(total_features_without_zero):
            if len(needed_cols) >= top_k:
                break
            if self.feature_importances.iloc[i]['importance_gain'] < CONSTANT.MIN_FEAT_IMPORTANT:
                break
            cur_col = self.feature_importances.iloc[i]['features']
            if cur_col in needed_cols:
                continue
            needed_cols.append(cur_col)

        self.ops['low_importance'] = list(set(list(X_train.columns)) - set(needed_cols))
        self.ops['needed_cols']=needed_cols
        #for c in self.ops['low_importance']:
        #    print(c)
        log('%d features with low importance.\n' % len(self.ops['low_importance']))

    @timeit
    def hyperopt_cat_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                          params: Dict, topk=6, free_cat_list=[]):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "scale_pos_weight": hp.uniform('x', 0, 5),
            "max_cat_group ": hp.choice("max_cat_group", np.linspace(0, 2000, 20, dtype=int)),
            "max_cat_threshold": hp.choice("max_cat_threshold", np.linspace(0, 2000, 20, dtype=int)),
            "cat_smooth": hp.choice("cat_smooth", np.linspace(0, 10000, 20, dtype=int)),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 500,
                              valid_data, early_stopping_rounds=100, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            feature_importance_df = pd.DataFrame()
            feature_importance_df["features"] = X_train.columns
            feature_importance_df["importance_gain"] = model.feature_importance(importance_type='gain')

            # in classification, less is better
            return {'loss': -score, 'status': STATUS_OK, "sort_feats": feature_importance_df, "best_iter": model.best_iteration}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=10, verbose=1,
                             rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"hyperopt auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        new_cat_feats = pd.DataFrame([])

        for result in trials.results:
            feature_import = result['sort_feats'].set_index('features')
            feature_import.drop(free_cat_list, axis=0, inplace=True, errors='ignore')
            if len(new_cat_feats) == 0:
                new_cat_feats = feature_import
            else:
                new_cat_feats = new_cat_feats + feature_import

        new_cat_feats = new_cat_feats[new_cat_feats['importance_gain'] > CONSTANT.MIN_CAT_FEAT_IMPORTANT * max(len(trials.results)/ 2, 1)]
        new_cat_feats.sort_values('importance_gain', inplace=True, ascending=False)
        if len(new_cat_feats) < topk:
            log(f"Test: test pass")
            return new_cat_feats.index.tolist(), hyperparams
        return new_cat_feats.index.tolist()[0:topk], hyperparams

    @timeit
    def identify_low_cat_importance(self, valid_ratio=0.2, topk=6, free_cat_list=[]):

        cat_feats = [col for col in self.data.columns if col.startswith(CONSTANT.CATEGORY_PREFIX) or col.startswith(CONSTANT.CATEGORY_HIGH_PREFIX)]
        if len(cat_feats) == 0:
            return []

        # self.data[cat_feats] = self.data[cat_feats].astype('category')
        # X_train, X_val, y_train, y_val = data_split(self.data[cat_feats], self.labels, valid_ratio)
        tmp_data = self.data[cat_feats].astype('category')
        X_train, X_val, y_train, y_val = data_split(tmp_data[cat_feats], self.labels, valid_ratio)

        print(f'x_train_len: {len(X_train)} x_val_len: {len(X_val)}')
        if len(CONSTANT.cat_params) == 0:
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "seed": 1,
                "num_threads": 4,
                "max_depth": 6,
                "num_leaves": 32,
                "feature_fraction": 0.6,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }

            save_cat_features, hyperparams = self.hyperopt_cat_lightgbm(X_train, y_train, X_val, y_val, params, topk, free_cat_list)
            CONSTANT.cat_params = {**params, **hyperparams}
            return save_cat_features
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(CONSTANT.cat_params,
                              train_data,
                              500,
                              valid_data,
                              early_stopping_rounds=100,
                              verbose_eval=100)
            feature_import = pd.DataFrame()
            feature_import["features"] = X_train.columns
            feature_import["importance_gain"] = model.feature_importance(importance_type='gain')
            feature_import.set_index('features', inplace=True)

            feature_import.drop(free_cat_list, axis=0, inplace=True, errors='ignore')
            feature_import = feature_import[feature_import['importance_gain'] > CONSTANT.MIN_CAT_FEAT_IMPORTANT]
            feature_import.sort_values('importance_gain', inplace=True, ascending=False)
            if len(feature_import) < topk:
                return feature_import.index.tolist()
            return feature_import.index.tolist()[0:topk]
