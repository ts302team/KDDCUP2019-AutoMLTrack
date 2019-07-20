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
from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from util import Config, log, timeit
from preprocess import *


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config, timer):
    return train_lightgbm(X, y, config, timer)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)
    return preds


@timeit
def validate(preds, y_path) -> np.float64:
    score = roc_auc_score(pd.read_csv(y_path)['label'].values, preds)
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config, timer):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    # 下面给出一个模型融合的例子
    predict_time = 0
    marjoy_has_sample_index = []  # 已经采样过的多数类索引
    seed = 1029

    max_train_time = 0

    is_binary = True
    use_all_data = False
    bset_iters = []
    print(f"est predict time: {predict_time}")

    X, X_val, y, y_val = time_series_data_split(X, y, config['time_series_index'], max_val_size=30000)
    log(f"time series split, train_size {len(y)}, val_size: {len(y_val)}")
    X_train, y_train, majority_index, isfull = down_sample_fast(X, y, 5.0, marjoy_has_sample_index)
    print(f"{len(y_train)}, {len(y_val)}")
    X_train_sample, y_train_sample = data_sample_for_train(X_train, y_train, 40000, 'train')
    hyperparams, drop_feature, beset_iter = hyperopt_lightgbm(X_train_sample, y_train_sample, X_val,
                                                              y_val, params, config)
    bset_iters.append(beset_iter)

    while True:
        gc.collect()
        start_time = time.time()
        if is_binary:
            params["objective"] = "binary"
            is_binary = False
        else:
            params["objective"] = "regression"
            is_binary = True

        seed_everything(seed)
        seed = (seed * 2) % (2 ** 32 - 1)

        if isfull:
            marjoy_has_sample_index = []
        marjoy_has_sample_index.extend(majority_index)

        gc.collect()
        if use_all_data:
            train_data = lgb.Dataset(X_train, label=y_train)
            iter = int(np.mean(bset_iters))
            log(f"train full data: after {iter} iters stop")
            model = lgb.train({**params, **hyperparams},
                              train_data,
                              iter,
                              verbose_eval=100)
            use_all_data = False
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train({**params, **hyperparams},
                              train_data,
                              1000,
                              valid_data,
                              early_stopping_rounds=100,
                              verbose_eval=100)
            bset_iters.append(model.best_iteration)

        end_time = time.time()
        train_time = end_time - start_time  # 计算模型训练时间和超参数提取时间
        max_train_time = max(max_train_time, train_time)

        predict_time += max_train_time / 2  # 此处简单的设置该模型预测的时间为训练时间的1/2， 这里应该更加的严谨【待考虑】
        next_train_time = max_train_time  # 并且预估下一次模型训练的时间同最长模型的训练时间
        next_predict_time = max_train_time / 2  # 并且预估下一次模型预测的时间同当前模型的预测时间
        still_has_time = predict_time + next_train_time + next_predict_time  # 如果再训练一个模型，仍然需要的时间：所有跑出来的模型的预测时间+预估下一个模型训练时间+预估下一个模型测试的时间

        try:
            config["model"].append(model)
        except:
            config["model"] = []
            config["model"].append(model)

        remain_time = TimeManager.get_time_left()
        print(f"est predict time: {predict_time}, est next train time: {next_train_time}, "
              f"est next predict time: {next_predict_time}, total est time: {still_has_time}, remain time: {remain_time}")
        if still_has_time >= remain_time or len(config["model"]) >= CONSTANT.MAX_MODEL_SIZE:
            break

        if remain_time - still_has_time < next_train_time or len(config["model"]) == CONSTANT.MAX_MODEL_SIZE-1:
            use_all_data = True

        if use_all_data:
            X_train, y_train, majority_index, isfull = down_sample_fast(
                pd.concat([X, X_val], axis=0, ignore_index=True), pd.concat([y, y_val], axis=0, ignore_index=True), 5.0, [])
        else:
            X_train, y_train, majority_index, isfull = down_sample_fast(X, y, 5.0, marjoy_has_sample_index)

        # uncomment this break to train only once
        # break

    print("model length:", len(config["model"]))

    feature_importance_df = pd.DataFrame()
    feature_importance_df["features"] = X.columns.tolist()
    feature_importance_df["importance_gain"] = model.feature_importance(importance_type='gain')
    feature_importance_df["importance_split"] = model.feature_importance(importance_type='split')
    return feature_importance_df


@timeit
def min_max_standard(predict):
    def help(x):
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    for i in range(len(predict)):
        predict[i] = help(predict[i])
    return predict


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    result = np.zeros(len(X))
    result_list = []
    time_remain = TimeManager.get_time_left()

    max_predict_time = 0
    for i in range(len(config["model"])):

        start_time = time.time()

        gc.collect()
        model = config["model"][i]
        res = model.predict(X)
        res = min_max_standard(res)
        result_list.append(res)

        end_time = time.time()
        print("predict round : ", i, " finish")
        max_predict_time = max(max_predict_time, end_time - start_time)
        time_remain -= (end_time - start_time)
        if max_predict_time * 2 > time_remain:
            break

    m = len(result_list)
    for i in range(m):
        result += (result_list[i] / m)

    print("result length:", m)

    if m == 0:
        print("predict_lightgbm function error : zero model")
        return min_max_standard(config["model"][0].predict(X))

    return result


@timeit
def hyperopt_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict,
                      config: Config):

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        #"max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(16, 64, 4, dtype=int)),
        #"feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        #"bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        #"bagging_freq": hp.choice("bagging_freq", np.linspace(0, 10, 1, dtype=int)),
        # "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        # "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        # "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        # "scale_pos_weight": hp.uniform('x', 0, 5),
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
    return hyperparams, drop_feature, trials.best_trial['result']['best_iter']
