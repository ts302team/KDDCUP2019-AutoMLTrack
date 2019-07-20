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
from util import log, timeit
import collections
from sklearn.model_selection import KFold
from numpy.random import normal

from encoders import *
from feature_selector import *
from merge import *
from CONSTANT import *
from resource_manager import TimeManager
import collections

class PreprocessClass:

    def __init__(self):
        self.encoder_objs = collections.OrderedDict({MAIN_TABLE_NAME: {}})

    @timeit
    def transform_categorical_encode(self, table_name, df):
        category_list = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
        for c in category_list:
            name = getName(CONSTANT.CAT_FRE_PREFIX, c, 'Frequency_Cat', table_name=table_name)
            obj = FrequencyCatEncoder({'cols': [c], 'name': name, 'all_rows': CONSTANT.TABLE_LENGTHS[table_name],
                                       'sample_rows': len(df.index)})
            obj.fit_transform(df)
            try:
                self.encoder_objs[table_name][name] = obj
            except KeyError:
                self.encoder_objs[table_name] = {}
                self.encoder_objs[table_name][name] = obj

    @timeit
    def fillna(self, df, isTrain=True, only_one_row=False):
        for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
            df[c].fillna(0, inplace=True)
            df[c] = pd.to_numeric(df[c], downcast='float')

        for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
            df[c].fillna(-1, inplace=True)
            df[c] = pd.to_numeric(df[c], downcast='signed')

        for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
            df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

        if only_one_row:
            for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
                df[c].fillna("0", inplace=True)
        else:
            df.drop([c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)], axis=1,inplace=True)

        for c in [c for c in df if c.startswith(CONSTANT.CAT_INT_PREFIX)]:
            df[c].fillna(0, inplace=True)
            df[c] = pd.to_numeric(df[c], downcast='unsigned')

    @timeit
    def transform_multicat_encode(self, table_name, df):
        category_list = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
        for c in category_list:
            name = getName(f'{CONSTANT.MULTI_CAT_FRE_PREFIX}', c, 'transform_multicat_encode', table_name)
            params = {'cols': [c], 'name': name, 'all_rows': CONSTANT.TABLE_LENGTHS[table_name],
                      'sample_rows': len(df.index)}
            obj = FrequencyMulcatEncoder(params)
            obj.fit_transform(df)
            try:
                self.encoder_objs[table_name][name] = obj
            except KeyError:
                self.encoder_objs[table_name] = {}
                self.encoder_objs[table_name][name] = obj

    @timeit
    def transform_datetime(self, table_name, df):
        for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
            name = getName(CONSTANT.CAT_TIME_PREFIX, c, 'transform_datetime', table_name)
            params = {'name': name, 'cols': [c],'all_rows': CONSTANT.TABLE_LENGTHS[table_name],
                      'sample_rows': len(df.index)}
            obj = DatatimeEncoder(params)
            obj.fit_transform(df)
            try:
                self.encoder_objs[table_name][name] = obj
            except KeyError:
                self.encoder_objs[table_name] = {}
                self.encoder_objs[table_name][name] = obj

    @timeit
    def transform_multicat_hash(self, table_name, df):
        for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            name = getName(CONSTANT.NUMERICAL_PREFIX, c, 'transform_multicat_hash', table_name)
            params = {'name': name, 'cols': [c],'all_rows': CONSTANT.TABLE_LENGTHS[table_name],
                      'sample_rows': len(df.index)}
            obj = BinaryEncoder(params)
            obj.fit_transform(df)
            try:
                self.encoder_objs[table_name][name] = obj
            except KeyError:
                self.encoder_objs[table_name] = {}
                self.encoder_objs[table_name][name] = obj

    @staticmethod
    def get_drop_features(df):
        return [c for c in df.columns if
                         c.startswith(CONSTANT.CATEGORY_PREFIX) or c.startswith(CONSTANT.MULTI_CAT_PREFIX) or \
                         c.startswith(CONSTANT.TIME_PREFIX) or c.startswith(CONSTANT.CATEGORY_HIGH_PREFIX)]

    @timeit
    def drop_features(self, df):
        gc.collect()
        drop_features = PreprocessClass.get_drop_features(df)
        if len(drop_features):
            try:
                df.drop(drop_features, axis=1, inplace=True)
            except Exception:
                print("drop features error!")

    @timeit
    def drop_mulcat_features(self, df):
        gc.collect()
        drop_features = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
        if len(drop_features):
            try:
                df.drop(drop_features, axis=1, inplace=True)
            except Exception:
                print("drop mulcat features error!")

    @timeit
    def transformer_ctr_encode(self, X, y, table_name=MAIN_TABLE_NAME):
        type = [c for c in X if (c.startswith(CONSTANT.CATEGORY_PREFIX) or \
                c.startswith(CONSTANT.CATEGORY_HIGH_PREFIX) or c.startswith(CONSTANT.CAT_FRE_PREFIX)) \
                and c not in CONSTANT.ctr_has_done and (c not in CONSTANT.REVERSE_NAME_MAP or CONSTANT.REVERSE_NAME_MAP[c] not in CONSTANT.ctr_has_done)]

        name = getName(CONSTANT.NUMERICAL_PREFIX, '', 'transformer_ctr_encode')
        params = {'name': name, 'all_rows':CONSTANT.TABLE_LENGTHS[table_name], 'sample_rows': len(X.index), 'type': type}
        obj = CTREncode(params)
        obj.fit_transform(X, y)
        try:
            self.encoder_objs[table_name][name] = obj
        except KeyError:
            self.encoder_objs[table_name] = collections.OrderedDict()
            self.encoder_objs[table_name][name] = obj

        CONSTANT.ctr_has_done = CONSTANT.ctr_has_done | set(type)

    @timeit
    def second_order_count_encode(self, X, y, visited_features=set(), table_name=MAIN_TABLE_NAME):
        category_list = [c for c in X if c.startswith(CONSTANT.CATEGORY_PREFIX)]
        number_of_encode = 0
        for i in range(len(category_list)):
            for j in range(i+1, len(category_list)):
                key = category_list[i]
                other = category_list[j]
                if key == other:
                    continue
                name = getName(CONSTANT.NUMERICAL_PREFIX, [key, other], 'second_order_count_encode')
                if name in CONSTANT.count_encoder_has_done:
                    continue
                CONSTANT.count_encoder_has_done.add(name)

                params = {'name': name, 'cols': [key, other], 'all_rows':CONSTANT.TABLE_LENGTHS[table_name],
                          'sample_rows': len(X.index)}
                obj = SecondOrderCountEncoder(params)
                obj.fit_transform(X, y)
                #self.encoder_objs[table_name][name] = obj
                try:
                    self.encoder_objs[table_name][name] = obj
                except KeyError:
                    self.encoder_objs[table_name] = collections.OrderedDict()
                    self.encoder_objs[table_name][name] = obj

                number_of_encode += 1
                if number_of_encode >= CONSTANT.MAX_UNIQUE_ENCODE:
                    break

    @timeit
    def second_order_nunique_encode(self, X, y=None, table_name=MAIN_TABLE_NAME):
        cat_list = [c for c in X if c.startswith(CONSTANT.CATEGORY_PREFIX) or c.startswith(CONSTANT.CATEGORY_HIGH_PREFIX)]
        number_of_encode = 0
        for key in cat_list:
            try:
                other_list = list(set(cat_list) - CONSTANT.unique_cat_has_done[key])
            except:
                CONSTANT.unique_cat_has_done[key] = set()
                other_list = copy.deepcopy(cat_list)

            if len(other_list):
                name = getName(CONSTANT.CAT_UNI_PREFIX, key, 'second_order_nunique_encode')
                params = {'name': name, 'key': key, 'type': other_list, 'all_rows': CONSTANT.TABLE_LENGTHS[table_name],
                          'sample_rows': len(X.index)}
                obj = SecondOrderNuniqueEncoder(params)
                obj.fit_transform(X)
                #self.encoder_objs[table_name][name] = obj
                try:
                    self.encoder_objs[table_name][name] = obj
                except KeyError:
                    self.encoder_objs[table_name] = collections.OrderedDict()
                    self.encoder_objs[table_name][name] = obj

                CONSTANT.unique_cat_has_done[key] = CONSTANT.unique_cat_has_done[key] | set(other_list)
                number_of_encode += len(other_list)
                if number_of_encode >= CONSTANT.MAX_UNIQUE_ENCODE:
                    break

    @timeit
    def second_order_cat_cat_encode(self, X, y, table_name=MAIN_TABLE_NAME):
        #total_cat_list = [c for c in X.columns if c.startswith(CONSTANT.CATEGORY_PREFIX) or c.startswith(CONSTANT.CATEGORY_HIGH_PREFIX)]
        cat_list = [c for c in X.columns if c.startswith(CONSTANT.CATEGORY_PREFIX) or c.startswith(CONSTANT.CATEGORY_HIGH_PREFIX)]
        origin_columns = set(X.columns.tolist())
        new_encoder_objs = {}

        number_of_encode = 0
        for i in range(len(cat_list)):
            for j in range(i+1, len(cat_list)):
                key = cat_list[i]
                other = cat_list[j]

                name = getName(CONSTANT.CATEGORY_HIGH_PREFIX, [key, other], 'second_order_cat_cat_encode')
                if name in CONSTANT.second_cat2cat_has_done:
                    continue
                CONSTANT.second_cat2cat_has_done.add(name)
                number_of_encode += 1

                params = {'name': name, 'key': key, 'other': other, 'all_rows': CONSTANT.TABLE_LENGTHS[table_name],
                          'sample_rows': len(X.index)}
                obj = SecondOrderCatCatEncoder(params)
                obj.fit_transform(X)
                new_encoder_objs[name] = obj

                if number_of_encode >= CONSTANT.MAX_CAT_CAT_ENCODE:
                    break

        new_columns = set(X.columns.tolist())
        fs = FeatureSelector(data=X, labels=y)
        new_cat = fs.identify_low_cat_importance(topk=TimeManager.time_budget//100, free_cat_list = cat_list) #默认的二阶cat类型数量
        drop_columns = new_columns - origin_columns - set(new_cat)

        for name in new_cat:
            try:
                self.encoder_objs[table_name][name] = new_encoder_objs[name]
            except:
                self.encoder_objs[table_name] = collections.OrderedDict()
                self.encoder_objs[table_name][name] = new_encoder_objs[name]
        if len(drop_columns):
            X.drop(drop_columns, axis=1, inplace=True)

        del new_encoder_objs
        return new_cat

    @timeit
    def transform_categorical_encode_for_high_order_cat(self, X, y=None, table_name=MAIN_TABLE_NAME):
        category_list = [c for c in X if c.startswith(CONSTANT.CATEGORY_HIGH_PREFIX) \
                         and c not in CONSTANT.fre_has_done and (c not in CONSTANT.REVERSE_NAME_MAP or CONSTANT.REVERSE_NAME_MAP[c] not in CONSTANT.fre_has_done)]
        #print("**************************transform_categorical_encode_for_high_order_cat*******************************")
        #print(category_list)
        #category_list = list(set(category_list) - CONSTANT.fre_has_done)

        for c in category_list:
            name = getName(CONSTANT.CAT_FRE_PREFIX, c, 'Frequency_Cat', table_name=table_name)
            obj = FrequencyCatEncoder({'cols': [c], 'name': name, 'all_rows': CONSTANT.TABLE_LENGTHS[table_name],
                                       'sample_rows': len(X.index)})
            obj.fit_transform(X)
            try:
                self.encoder_objs[table_name][name] = obj
            except KeyError:
                self.encoder_objs[table_name] = collections.OrderedDict()
                self.encoder_objs[table_name][name] = obj

        CONSTANT.fre_has_done = CONSTANT.fre_has_done | set(category_list)

# 大表处理
class FeatureIteration:
    def __init__(self, config, params=None):
        self.drop_table = {}
        self.ops = {}
        self.index = 0
        self.config = config
        self.time_cost_ratio=MERGED_TABLE_TIME_RATIO
        self.free_cols = []
        self.needed_cols=None
        self.top_k=-1
        self.hyperparams = params
        self.origin_feats_num = 0

    @timeit
    def featrue_selection(self, X, y, top_k=200):
        '''
        特征选择需要做实验测试
        '''
        cur_drop_table = []

        if self.index % 4 < 2:
            top_k = X.shape[1]
        elif self.index % 4 == 2:
            top_k = int(len(self.free_cols) + TimeManager.time_budget / 100)
        else:
            top_k = min(int(len(self.free_cols) + min(TimeManager.time_budget / 10, 30)), self.bound)

        fs = FeatureSelector(data=X, labels=y)
        fs.identify_collinear(correlation_threshold=1)   # 如果产生共线特征，肯定是新做的特征与旧的特征或者新做的特征与新的特征，返回的都是新的特征

        cur_drop_table += fs.ops['collinear']
        #print(fs.ops['collinear'])

        fs.identify_low_importance(top_k, free_list=self.free_cols, params=self.hyperparams, not_need_list = fs.ops['collinear'])    # 如果将共线特征加入到lgbm进行选择，会影响有用的特征
        cur_drop_table += fs.ops['low_importance']
        #print(fs.ops['low_importance'])

        # -------------------zyp for time control ------------------------------
        self.needed_cols = fs.ops['needed_cols']
        # ---------------------------------------- ------------------------------

        del fs
        return set(cur_drop_table), len(self.needed_cols)  # 这个地方返回的top_k有问题，是不是应该返回self.needed_cols的长度

    @timeit
    def feature_engineering(self, X, y=None):
        '''
        高阶特征放在这里
        '''

        if self.origin_feats_num > 0 and len(X.columns.tolist()) >= max(self.origin_feats_num * CONSTANT.MAX_SECOND_RATE, MIN_NUM_SECOND_FEATURES):
            log(f"current number of features is {len(X.columns.tolist())}, beyond {max(self.origin_feats_num * CONSTANT.MAX_SECOND_RATE, MIN_NUM_SECOND_FEATURES)}")
            return False

        if self.index >= CONSTANT.MAX_FEATURE_ITERATIONS:
            log(f"byond max number of iterations")
            return False

        Name_Transform.fit_transform(X)
        self.free_cols = X.columns.tolist().copy()

        if self.index == 0:
            self.origin_feats_num = len(self.free_cols)
            self.bound = max(self.origin_feats_num * CONSTANT.MAX_SECOND_RATE, MIN_NUM_SECOND_FEATURES)

        print(f"current features : {len(self.free_cols)}")
        prep_class = PreprocessClass()

        if self.index % 4 == 0:
            prep_class.second_order_cat_cat_encode(X, y) #产生新的cat
            self.free_cols = X.columns.tolist().copy()
            prep_class.transform_categorical_encode_for_high_order_cat(X, y)
        elif self.index % 4 == 1:
            prep_class.transformer_ctr_encode(X, y)
        elif self.index % 4 == 2:
            prep_class.second_order_nunique_encode(X, y)
        else:
            prep_class.second_order_count_encode(X, y)

        cur_drop_table, self.top_k = self.featrue_selection(X, y)
        #new_num_feats = set(X.columns.tolist().copy()) - cur_drop_table - set(self.free_cols)  # 要保存的列

        if self.config['time_col'] in cur_drop_table:
            cur_drop_table.remove(self.config['time_col'])

        try:
            self.modify_map(X, cur_drop_table, prep_class.encoder_objs)
        except Exception:
            pass

        print(f"topk: {X.shape[1]} == > {self.top_k}")
        if len(set(X.columns.tolist()) - set(self.free_cols)) == 0 and self.index % 4 == 3:
            log(f"no features has been producted!")
            return False

        # ********************* JUST FOR DEBUG ***************************** #
        #print('******************** needed_cols *******************8')
        #for col in self.needed_cols:
        #    if 'second_' in col:
        #        print(col)
        #print('******************** needed_cols *******************8')

        #print('******************** encoders *******************8')
        #for i in range(self.index):
        #for table_name, func_list in self.ops[self.index-1].items():
        #    for func_name, func in func_list.items():
        #        if 'second_' in func_name:
        #            print(func_name)
        #            try:
        #                for t in func.type:
        #                    print(f'{func_name}#{t}')
        #            except:
        #                pass
        #print('******************** encoders *******************8')
        # ********************* JUST FOR DEBUG ***************************** #
        return True

    #@staticmethod
    def drop_objs(self,cur_drop_table,encoder_objs,table_name=MAIN_TABLE_NAME,test=False):
        number = 0
        for name in cur_drop_table:
            if '#' in name:
                tmp = name.split('#', 1)
                name = tmp[0]
                type = tmp[1]
                #if test:
                #    print(type in self.ops[self.index-1][table_name][name].type)
                try:
                    encoder_objs[table_name][name].type.remove(type)
                    encoder_objs[table_name][name].remove_type_time(type)
                    number += 1
                except:
                    pass
                #if test:
                #    print(type in self.ops[self.index-1][table_name][name].type)
                #    exit()
                if name in encoder_objs[table_name]:
                    type_len=len(encoder_objs[table_name][name].type)
                    if type_len == 0 or type_len==1 and (encoder_objs[table_name][name].type[0]+'_' in name):
                        encoder_objs[table_name].pop(name)
            else:
                try:
                    encoder_objs[table_name].pop(name)
                    number += 1
                except:
                    pass
        print(f"drop features number: {number} ==> {len(cur_drop_table)}")

    def top_k_modify_map(self, X, top_k):
        if top_k==-1:
            return
        drop_table=self.needed_cols[top_k:]
        self.drop_objs(drop_table, self.ops[self.index-1],MAIN_TABLE_NAME)
        self.drop_table[self.index-1] |= set(drop_table)
        if len(drop_table):
            X.drop(drop_table, axis=1, inplace=True)

    def get_top_k_drop_cols(self,top_k):
        return self.needed_cols[top_k:]

    @timeit
    def modify_map(self, X, cur_drop_table, encoder_objs, table_name=MAIN_TABLE_NAME):
        self.drop_objs(cur_drop_table, encoder_objs)
        self.drop_table[self.index] = cur_drop_table
        if len(cur_drop_table):
            X.drop(cur_drop_table, axis=1, inplace=True)
        self.ops[self.index] = encoder_objs
        self.index += 1


    @timeit
    def feature_resume(self, X, y=None, isTrain=True):
        # ------------------------time control comparison info ---------------------------------
        #if isTrain:
        #    print("merged table time estimate: "+str(self.get_estimated_time_for_all_data()))
        # ---------------------------------------------------------------------------------------
        drop_features = PreprocessClass.get_drop_features(X)
        extra_sub_mem = 0
        try:
            extra_sub_mem = X[drop_features].memory_usage().sum()
        except:
            pass

        timer = Timer()
        resource_break = False
        for i in range(self.index):
            Name_Transform.transform(X)

            origin_feature_n = X.shape[1]
            #for table_name, func_list in self.ops[i].items():
            #    for func_name, func in func_list.items():
            #        print(func_name)
            for table_name, func_list in self.ops[i].items():
                for func_name, func in func_list.items():
                    try:
                        func.fit_transform(X, y)
                    except Exception:
                        print(f"error: feture resume error, function name: {func_name}")

                    if not MemoryManager.simple_check_mem(X, y, extra_sub_mem):
                        resource_break = True
                        print("********* feature resume mem break **************")
                        break
                    if not TimeManager.simple_check_time():
                        resource_break = True
                        print("********* feature resume time break **************")
                        break
                if resource_break:
                    break

            n_feats = X.shape[1] - origin_feature_n
            drop_columns = self.drop_table[i].intersection(set(X.columns))  # 删除一部分列，这些列存在的原因是，有些更有用的列可能基于之前无用的列生成

            timer.check(f"iter {i}: {n_feats} features has resumed")
            try:
                if len(drop_columns):
                    X.drop(drop_columns, axis=1, inplace=True)
            except Exception:
                pass
            n_feats = X.shape[1] - origin_feature_n
            timer.check(f"iter {i}: {n_feats} features has resumed after drop")

            if resource_break:
                break

    def get_estimated_time_for_all_data(self, top_k=-1):
        total_time = 0
        if top_k == -1:
            for i in range(self.index):
                for table_name, func_list in self.ops[i].items():
                    for func_name, func in func_list.items():
                            total_time += func.get_all_data_time_estimate()
            print(f'total time all: {total_time}')

        else:
            for i in range(self.index-1):
                for table_name, func_list in self.ops[i].items():
                    for func_name, func in func_list.items():
                        total_time += func.get_all_data_time_estimate()

            # **************** 注意！！ 这里默认二阶特征名字中均含有second_ ****************************
            selected_features = self.needed_cols[len(self.free_cols):top_k]
            # **************** **************************************** ****************************
            print(f'self index{self.index}')
            encoder_objs=self.ops[self.index-1]
            table_name = MAIN_TABLE_NAME

            func_name_set = set([name.split('#', 1)[0] for name in selected_features if '#' in name])
            for name in func_name_set:
                try:
                    total_time += encoder_objs[table_name][name].get_all_data_time_estimate('extra')
                except:
                    log(f"error: can not find {name} in encoder_objs")

            func_name_set.clear()
            for name in selected_features:
                print(name)
                if '#' in name:
                    tmp = name.split('#', 1)
                    try:
                        total_time += encoder_objs[table_name][tmp[0]].get_all_data_time_estimate(tmp[1])
                    except:
                        func_name_set.add(tmp[0])
                else:
                    try:
                        total_time += encoder_objs[table_name][name].get_all_data_time_estimate()
                    except:
                        log(f"error: can not find {name} in encoder_objs")

            for name in func_name_set:
                try:
                    total_time += encoder_objs[table_name][name].get_all_data_time_estimate('all')
                except:
                    log(f"error: can not find {name} in encoder_objs")

            print(f'total time with 2nd: {total_time}')

        return total_time/self.time_cost_ratio


# 分表处理
class FeatureIterationXs(FeatureIteration):
    def __init__(self, config):
        super().__init__(config)
        self.ops = None
        self.drop_table = None
        self.time_cost_ratio=SEPARATE_TABLE_TIME_RATIO

    @timeit
    def feature_engineering(self, Xs, y=None):
        print('main : ', Xs[MAIN_TABLE_NAME].shape)
        print('y : ', len(y))
        prep_class = PreprocessClass()
        # 分表做frequecy等编码
        for name, data in Xs.items():
            prep_class.transform_categorical_encode(name, data)
            prep_class.transform_multicat_hash(name, data)
            prep_class.transform_multicat_encode(name, data)
            prep_class.transform_datetime(name, data)

        try:
            X = merge_table(Xs, self.config)
        except Exception:
            X = Xs[MAIN_TABLE_NAME].copy()

        print('FeatureIterationXs after merge : ', X.shape)
        prep_class.fillna(X)

        cur_drop_table = self.featrue_selection(X, y)
        cur_drop_table = set(col for col in cur_drop_table if "encode" in col)
        self.modify_map(X, cur_drop_table, prep_class.encoder_objs)

        return X

    @timeit
    def featrue_selection(self, X, y):
        cur_drop_table = []
        fs = FeatureSelector(data=X, labels=y)
        fs.identify_collinear(correlation_threshold=0.999)
        cur_drop_table += fs.ops['collinear']

        self.hyperparams = fs.identify_zero_importance(not_need_list=fs.ops['collinear'])
        cur_drop_table += fs.ops['zero_importance']
        del fs
        return set(cur_drop_table)

    @timeit
    def modify_map(self, X, cur_drop_table, encoder_objs, table_name=None):
        # 改正encode_objs，改正X
        needed_cols = list(X.columns)
        for name in cur_drop_table:
            needed_cols.remove(name)

        def get_original_col(name):
            if '(' in name:
                name = ")".join(name.split('(', 1)[1].split(')', -1)[:-1])
                needed_cols.append(name)
                get_original_col(name)
            if '#' in name:
                name = name.split('#')[0]
                needed_cols.append(name)
                get_original_col(name)

        for name in needed_cols:
            get_original_col(name)

        drop_number = 0

        new_encoders = {}
        for table_name in encoder_objs.keys():
            new_encoders[table_name] = {}
            for col_name in encoder_objs[table_name].keys():
                if col_name not in needed_cols:
                    try:
                        drop_number += len(encoder_objs[table_name][col_name].type)
                    except Exception:
                        drop_number += 1
                    continue

                new_encoders[table_name][col_name] = encoder_objs[table_name][col_name]
                try:
                    col_type = encoder_objs[table_name][col_name].type
                    remove_type = []
                    for t in col_type:
                        name = f'{col_name}#{t}'
                        if name not in needed_cols:
                            remove_type.append(t)
                    for t in remove_type:
                        new_encoders[table_name][col_name].type.remove(t)
                        drop_number += 1
                except:
                    pass

        del encoder_objs
        self.drop_table = cur_drop_table
        if len(cur_drop_table):
            X.drop(cur_drop_table, axis=1, inplace=True)
        self.ops = new_encoders
        print(f"drop features number: {drop_number} ==> {len(cur_drop_table)}")

    @timeit
    def feature_resume(self, Xs, y=None, isTrain=True):

        timer = Timer()
        n_feats = 0
        for table_name, func_list in self.ops.items():
            if not isTrain and table_name != MAIN_TABLE_NAME:
                continue

            origin_features_n = Xs[table_name].shape[1]
            for func_name, func in func_list.items():
                try:
                    func.fit_transform(Xs[table_name])
                except Exception:
                    print(f"feature resume: not find {func_name}")
                    pass

            n_feats += (Xs[table_name].shape[1] - origin_features_n)

        # ------------------------ time control comparison info -------------------------------------
        if isTrain:
            print("separate table estimated time: "+str(self.get_estimated_time_for_all_data()))
        # --------------------------------------------------------------------------------------------

        timer.check(f"Xs: {n_feats} features has resumed")

        try:
            X = merge_table(Xs, self.config)
        except Exception:
            X = Xs[MAIN_TABLE_NAME].copy()

        #if not isTrain:  #如果为验证，则删除Xs
        #    del Xs
        #    gc.collect()

        PreprocessClass().fillna(X, isTrain=isTrain)  # merge完之后，可能会生成NaN的数据，所以在这一步最好加上填充空值，空值会影响后面的特征工程 （再商议）
        # 这一步之后也可以调用convert_type降低内存空间，不过需要时间，也可以在merge的时候就转化表格数据来优化空间，这里在余注意一下，可以进行优化

        drop_columns = self.drop_table.intersection(set(X.columns))  # 删除一部分列，这些列存在的原因是，表合并之后，有些有用的列可能基于无用的列生成
        try:
            if len(drop_columns):
                X.drop(drop_columns, axis=1, inplace=True)
        except Exception:
            pass
        return X

    def get_estimated_time_for_all_data(self):
        total_time = 0
        for table_name, func_list in self.ops.items():
            for func_name, func in func_list.items():
                total_time += func.get_all_data_time_estimate()
        return total_time/self.time_cost_ratio


class CatConvert:
    def __init__(self):
        self.cat_map = {}
        self.size = 0

    def map_info(self, x):
        if isNaN(x):
            return -1
        try:
            return self.cat_map[x]
        except:
            self.cat_map[x] = self.size
            self.size += 1
            return self.cat_map[x]

    def fit_transform(self, X):
        """
        zypang
        train data fit transform
        :param X: not quite intuitive. It actually is a col (pd.Series). Suggestion: df_col
        :return: converted col series.
        """
        # -----------zypang ------------------
        cat_val = X.astype('category')
        cat_dict_reverse = dict(enumerate(cat_val.cat.categories))
        self.cat_map = dict(zip(cat_dict_reverse.values(), cat_dict_reverse.keys()))
        self.size = len(self.cat_map)
        return cat_val.cat.codes #将0空出来，作为默认的填充值, 如果此处为全NULL，会不会报错阿
        # --------------------------------------

    def transform(self, X):
        return X.map(self.map_info)


class MulCatConvert:
    def __init__(self):
        self.mul_cat_map = {}
        self.size = 1

    def mc_convert(self, x):
        res = []
        try:
            for value in x.split(','):
                try:
                    res.append(self.mul_cat_map[value])
                except:
                    self.mul_cat_map[value] = str(self.size)
                    self.size += 1
                    res.append(self.mul_cat_map[value])
        except:
            res.append("0")
        return ",".join(set(res))

    def fit_transform(self, X):
        # tmp = X.values.tolist()
        # pset = set()
        # for values in tmp:
        #    for value in values.split(','):
        #        pset.add(value)
        # self.mul_cat_map = {label: str(idx) for idx, label in enumerate(pset)}
        # self.size = len(self.mul_cat_map)
        # del tmp
        return X.map(self.mc_convert)

    def transform(self, X):
        return X.map(self.mc_convert)


from sklearn.preprocessing import StandardScaler
class DataConvert:
    def __init__(self, col):
        self.col = col
        self.data_map = {}
        self.numer_mean = 0
        self.is_drop = False
        self.drop_rate = 0.999
        self.drop_threshold = 100

    def fit_transform(self, X):
        if self.col.startswith(CONSTANT.NUMERICAL_PREFIX):
            null_num = pd.isnull(X[self.col]).sum()
            total_num = len(X[self.col])
            if null_num / len(X[self.col]) > self.drop_rate and total_num - null_num < self.drop_threshold:
                log(f"drop column {self.col}")
                self.is_drop = True
                X.drop(self.col, axis=1, inplace=True)
                return

            self.numer_mean = X[self.col].mean()
            X[self.col].fillna(self.numer_mean, inplace=True)
            self.data_map = StandardScaler()
            X[self.col] = self.data_map.fit_transform(X[self.col].values.reshape(-1, 1))
            X[self.col] = pd.to_numeric(X[self.col], downcast='float')

        elif self.col.startswith(CONSTANT.CATEGORY_PREFIX):
            self.data_map = CatConvert()
            X[self.col] = self.data_map.fit_transform(X[self.col])

        elif self.col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            null_num = pd.isnull(X[self.col]).sum()
            total_num = len(X[self.col])
            if null_num / len(X[self.col]) > self.drop_rate and total_num - null_num < self.drop_threshold:
                log(f"drop column {self.col}")
                self.is_drop = True
                X.drop(self.col, axis=1, inplace=True)
                return

            self.data_map = MulCatConvert()
            X[self.col] = self.data_map.fit_transform(X[self.col])
        elif self.col.startswith(CONSTANT.TIME_PREFIX):
            delta = datetime.timedelta(microseconds=1)
            min_date = X[self.col].min()
            min_date = min_date - delta
            self.min_date = min_date
            X[self.col].fillna(self.min_date, inplace=True)

    def transform(self, X):
        if self.col.startswith(CONSTANT.NUMERICAL_PREFIX):
            if self.is_drop:
                log(f"drop column {self.col}")
                X.drop(self.col, axis=1, inplace=True)
                return

            X[self.col].fillna(self.numer_mean, inplace=True)
            X[self.col] = self.data_map.transform(X[self.col].values.reshape(-1, 1))
            X[self.col] = pd.to_numeric(X[self.col], downcast='float')

        elif self.col.startswith(CONSTANT.CATEGORY_PREFIX):
            X[self.col] = self.data_map.transform(X[self.col])

        elif self.col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            if self.is_drop:
                log(f"drop column {self.col}")
                X.drop(self.col, axis=1, inplace=True)
                return

            X[self.col] = self.data_map.transform(X[self.col])

        elif self.col.startswith(CONSTANT.TIME_PREFIX):
            delta = datetime.timedelta(microseconds=1)
            min_date = X[self.col].min()
            min_date = min_date - delta
            self.min_date = min_date
            X[self.col].fillna(self.min_date, inplace=True)
