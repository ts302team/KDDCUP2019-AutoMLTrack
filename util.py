import os
import time
from typing import Any

import CONSTANT
import random
import numpy as np
import pandas as pd

nesting_level = 0
is_start = None

class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")

class Config:
    def __init__(self, info):
        self.data = {
            **info,
            "start_time": time.time(),
        }
        self.data["tables"] = {}
        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    @staticmethod
    def aggregate_op(col):
        if CONSTANT.TOO_MANY_COLS:
            ops = {
                CONSTANT.NUMERICAL_TYPE: ["mean", "sum"],
                CONSTANT.CATEGORY_TYPE: [],
                CONSTANT.TIME_TYPE: [],
                CONSTANT.CAT_INT_TYPE: ["count"],
                CONSTANT.MULTI_CAT_TYPE: []
            }
        else:
            ops = {
                CONSTANT.NUMERICAL_TYPE: ["mean", "sum", "min", "max"],  # "count"],
                # CONSTANT.CATEGORY_TYPE: ["count", merge_cat],
                CONSTANT.CATEGORY_TYPE: [],  # "count"],
                CONSTANT.TIME_TYPE: ["max", "min"],
                CONSTANT.CAT_INT_TYPE: ["max", "min", "sum", "mean"],
                # CONSTANT.MULTI_CAT_TYPE: [merge_mult_cat, count_mult_cat],
                CONSTANT.MULTI_CAT_TYPE: []  # count_mult_cat],
            }
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            return ops[CONSTANT.CATEGORY_TYPE]
        if col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            #assert False, f"MultiCategory type feature's aggregate op are not supported."
            return ops[CONSTANT.MULTI_CAT_TYPE]
        if col.startswith(CONSTANT.TIME_PREFIX):
            return ops[CONSTANT.TIME_TYPE]
        if col.startswith(CONSTANT.CAT_INT_PREFIX):
            return ops[CONSTANT.CAT_INT_TYPE]
        #assert False, f"Unknown col type {col}"
        return []

    def time_left(self):
        return self["time_budget"] - (time.time() - self["start_time"])

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def getName(prefix, objects, func, table_name='', param=''):
    if isinstance(objects, list):
        return f'{prefix}encode_{table_name}{"_".join(objects)}_{func}_{param}'
    return f'{prefix}encode_{table_name}_{objects}_{func}_{param}'

@timeit
def down_sample_fast(X, y, frac=1.0, marjoy_has_sample_index=None, min_major_class_sample_size=20000):
    isfull = False

    MINORITY_THRESHOLD = 20000
    class_0_freq = len(y[y == 0])
    class_1_freq = len(y[y == 1])
    majority_class = 0
    if class_1_freq > class_0_freq:
        majority_class = 1
        minority_count = class_0_freq
    else:
        minority_count = class_1_freq

    minority_class = int(not majority_class)

    ### do downsampling as per remove percent ###

    indices = np.array(range(len(y)))
    majority_index = indices[y == majority_class]
    minority_index = indices[y == minority_class]

    if marjoy_has_sample_index is not None and len(marjoy_has_sample_index):
        majority_index = np.array(list(set(majority_index) - set(marjoy_has_sample_index)))
        if len(majority_index) < int(minority_count * frac):
            majority_index = indices[y == majority_class]
            isfull = True

    if int(minority_count * frac) > len(majority_index):
        size = len(majority_index)
    else:
        size = int(minority_count * frac)

    #if size < min_major_class_sample_size:# < len(majority_index): #加上此策略
    #    size = min(min_major_class_sample_size, len(majority_index))

    #majority_index = time_dense_choice(majority_index, size, list(range(80, 100, 1)))
    majority_index = np.random.choice(majority_index, size=size, replace=False)
    sorted_index = sorted(np.concatenate([minority_index, majority_index]))

    print('Sampled data size:', len(sorted_index))
    return X.iloc[sorted_index].reset_index(drop=True), y.iloc[sorted_index].reset_index(drop=True), majority_index, isfull


@timeit
def down_sample_for_feature_iteration(X, y, sample_size = 10000, min_minority_index_sample_size=200, magic_rate=5):
    """
    zypang
    changed some codes
    :param X:
    :param y:
    :param sample_size:
    :param min_minority_index_sample_size:
    :return:
    """
    total = len(y)
    class_1_freq = y.sum()
    class_0_freq = total -class_1_freq

    #zyp: majority_class = 0
    #zyp: if class_1_freq > class_0_freq:
    #zyp:     majority_class = 1

    majority_len = class_1_freq if class_1_freq > class_0_freq else class_0_freq
    minority_len = total-majority_len

    majority_class = int(class_1_freq > class_0_freq)
    minority_class = majority_class ^ 1

    ### do downsampling as per remove percent ###
    indices = np.array(range(len(y)))
    majority_index = indices[y == majority_class]
    minority_index = indices[y == minority_class]

    majority_ind_sample_size = int(sample_size/(magic_rate + 1) * magic_rate)
    minority_index_sample_size = sample_size - majority_ind_sample_size
    if minority_len < minority_index_sample_size:
        minority_index_sample_size = minority_len
        majority_ind_sample_size = minority_index_sample_size*magic_rate
    if majority_len < majority_ind_sample_size:
        majority_ind_sample_size = majority_len

    majority_index = np.random.choice(majority_index, size=majority_ind_sample_size, replace=False)
    minority_index = np.random.choice(minority_index, size=minority_index_sample_size, replace=False)
    sorted_index = sorted(np.concatenate([minority_index, majority_index]))

    return X.iloc[sorted_index].reset_index(drop=True), y.iloc[sorted_index].reset_index(drop=True)


def data_sample(X: pd.DataFrame, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    """
    zypang change to one line
    :param X:
    :param nrows:
    :return:
    """
    return X.copy() if len(X.index) <= nrows else X.sample(nrows, random_state=1).reset_index(drop=True)

def data_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    val_len = int(test_size * len(X))
    return X.iloc[:-val_len].reset_index(drop=True), X.iloc[-val_len:].reset_index(drop=True), \
           y.iloc[:-val_len].reset_index(drop=True), y.iloc[-val_len:].reset_index(drop=True)

def data_sample_for_train(X: pd.DataFrame, y: pd.Series, nrows: int = 5000, mode: str = 'train'):
    if len(X) > nrows:
        if mode == 'train':
            X_sample = X.iloc[-nrows:]
            y_sample = y.iloc[-nrows:]
        else:
            X_sample = X.iloc[0:nrows]
            y_sample = y.iloc[0:nrows]
    else:
        X_sample = X.copy()
        y_sample = y.copy()

    return X_sample, y_sample

def isNaN(num):
    return num != num

@timeit
def time_series_data_split(X, y, time_series_index, rate=0.2, max_rate=1.1, max_val_size=30000):
    max_sample_size = min(int(len(y) * rate), max_val_size)

    if len(time_series_index) > 1:
        try:
            total_index_set = set(y.index.tolist())
            n_len = len(time_series_index)
            valid_index = []
            for i in range(n_len-1, -1, -1):
                #if len(valid_index) + len(time_series_index[i]) > len(y) * max_rate:
                if len(valid_index) + len(time_series_index[i]) > max_sample_size * max_rate:
                    still_need = max_sample_size - len(valid_index)
                    valid_index.extend(time_series_index[i][-still_need:])
                    print("enter time_series_data_split rare path")
                    break

                valid_index.extend(time_series_index[i])
                if max_sample_size <= len(valid_index):
                    break

            valid_index = sorted(valid_index)
            train_index = list(total_index_set - set(valid_index))
            train_index = sorted(train_index)
            valid_index = list(valid_index)
            return X.iloc[train_index].reset_index(drop=True), X.iloc[valid_index].reset_index(drop=True), \
                   y.iloc[train_index].reset_index(drop=True), y.iloc[valid_index].reset_index(drop=True)
        except Exception:
            return X.iloc[:-max_sample_size].reset_index(drop=True), X.iloc[-max_sample_size:].reset_index(drop=True), \
                   y.iloc[:-max_sample_size].reset_index(drop=True), y.iloc[-max_sample_size:].reset_index(drop=True)

    return X.iloc[:-max_sample_size].reset_index(drop=True), X.iloc[-max_sample_size:].reset_index(drop=True), \
               y.iloc[:-max_sample_size].reset_index(drop=True), y.iloc[-max_sample_size:].reset_index(drop=True)

@timeit
def time_series_pre_data_sample(X, y, time_series_index, max_rate= 1.1, max_sample_size=1000000):
    if len(y) < max_sample_size:
        return X, y
    max_sample_size = int(min(len(y), max_sample_size))
    if len(time_series_index) > 1:
        try:
            n_len = len(time_series_index)

            sample_index = []
            for i in range(n_len-1, -1, -1):
                if len(sample_index) + len(time_series_index[i]) > max_sample_size * max_rate:
                    still_need = max_sample_size - len(sample_index)
                    sample_index.extend(time_series_index[i][-still_need:])
                    break

                sample_index.extend(time_series_index[i])
                if max_sample_size <= len(sample_index):
                    break

            sample_index = sorted(sample_index)
            sample_index = list(sample_index)
            return X.iloc[sample_index].reset_index(drop=True), y.iloc[sample_index].reset_index(drop=True)
        except Exception:
            return X.iloc[-max_sample_size:].reset_index(drop=True), y.iloc[-max_sample_size:].reset_index(drop=True)

    return X.iloc[-max_sample_size:].reset_index(drop=True), y.iloc[-max_sample_size:].reset_index(drop=True)

@timeit
def tables_data_sample(Xs, max_sample_size=300000):
    for name, data in Xs.items():
        if name == CONSTANT.MAIN_TABLE_NAME:
            continue
        if len(data) <= max_sample_size:
            continue
        indices = np.array(range(len(data)))
        index = np.random.choice(indices, size=max_sample_size, replace=False)
        Xs[name] = data.iloc[index].reset_index(drop=True)


def init_name_info():
    CONSTANT.NAME_MAP = {}
    CONSTANT.REVERSE_NAME_MAP = {}
    CONSTANT.CAT_UNI_INDEX = 0
    CONSTANT.CAT_FRE_INDEX = 0
    CONSTANT.NUM_INDEX = 0
    CONSTANT.TIME_INT_INDEX = 0
    CONSTANT.CATEGORY_HIGH_INDEX = 0
    CONSTANT.MULTI_CAT_FRE_INDEX = 0

def Second_Order_Info_Init():
    CONSTANT.fillna_map = {}
    CONSTANT.ctr_has_done = set()
    CONSTANT.second_cat2cat_has_done = set()
    CONSTANT.fre_has_done = set()
    CONSTANT.unique_cat_has_done = {}
    CONSTANT.time_cat_has_done = set()
    CONSTANT.cat_params = {
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
                "cat_smooth": 500,
                "max_cat_group": 100,
                "max_cat_threshold": 900,
            }
    CONSTANT.count_encoder_has_done = set()
    CONSTANT.TOO_MANY_COLS = False

def get_next_name_index(type):
    if type == CONSTANT.CAT_FRE_PREFIX:
        CONSTANT.CAT_FRE_INDEX += 1
        return CONSTANT.CAT_FRE_INDEX
    elif type == CONSTANT.CAT_UNI_PREFIX:
        CONSTANT.CAT_UNI_INDEX += 1
        return CONSTANT.CAT_UNI_INDEX
    elif type == CONSTANT.NUMERICAL_PREFIX:
        CONSTANT.NUM_INDEX += 1
        return CONSTANT.NUM_INDEX
    elif type == CONSTANT.CAT_TIME_PREFIX:
        CONSTANT.CAT_TIME_INDEX += 1
        return CONSTANT.CAT_TIME_INDEX
    elif type == CONSTANT.CATEGORY_HIGH_PREFIX:
        CONSTANT.CATEGORY_HIGH_INDEX += 1
        return CONSTANT.CATEGORY_HIGH_INDEX
    elif type == CONSTANT.MULTI_CAT_FRE_PREFIX:
        CONSTANT.MULTI_CAT_FRE_INDEX += 1
        return CONSTANT.MULTI_CAT_FRE_INDEX

class Name_Transform():
    def __init__(self):
        pass

    @staticmethod
    def fit_transform(X):
        feat_list = list(X.columns)
        new_feat_list = []
        for feat_name in feat_list:
            if (feat_name.endswith(CONSTANT.RENAME_SIGN) and "#" not in feat_name) or feat_name.startswith(CONSTANT.TIME_PREFIX) or feat_name.startswith(
                    CONSTANT.MULTI_CAT_PREFIX) or feat_name.startswith(CONSTANT.CATEGORY_PREFIX):
                new_feat_list.append(feat_name)
            else:
                type = feat_name.split('_', 1)[0] + "_"
                name_index = get_next_name_index(type)
                new_feat_name = type + str(name_index) + CONSTANT.RENAME_SIGN
                new_feat_list.append(new_feat_name)
                if feat_name in CONSTANT.NAME_MAP:
                    print(f"{feat_name}: {CONSTANT.NAME_MAP[feat_name]} write again")
                CONSTANT.NAME_MAP[feat_name] = new_feat_name
                CONSTANT.REVERSE_NAME_MAP[new_feat_name] = feat_name

        X.columns = new_feat_list

    @staticmethod
    def transform(X):
        feat_list = [c for c in X.columns]
        new_feat_list = []
        for feat_name in feat_list:
            if (feat_name.endswith(CONSTANT.RENAME_SIGN) and "#" not in feat_name) or feat_name.startswith(CONSTANT.TIME_PREFIX) or feat_name.startswith(
                    CONSTANT.MULTI_CAT_PREFIX) or feat_name.startswith(CONSTANT.CATEGORY_PREFIX):
                new_feat_list.append(feat_name)
            else:
                new_feat_list.append(CONSTANT.NAME_MAP[feat_name])
        X.columns = new_feat_list