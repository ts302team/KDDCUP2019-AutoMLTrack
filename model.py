import os
# ---- just for mem -------
#import tracemalloc
# ---- just for mem -------
os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")


from automl import predict, train
from preprocess import *
from util import *

class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.feat_drop = []
        self.prep_class = PreprocessClass()
        self.encoder_objs = {}
        self.drop_table_all = []
        self.dataCovert = {}
        self.table_relation = {}
        self.feature_iter_Xs = None
        self.feature_iter = None
        self.attrs2drop = None
        init_name_info()
        Second_Order_Info_Init()

    @timeit
    def convertX(self, Xs, config, is_train=True):
        '''
        优化空间，将大数映射为小数
        '''
        if is_train:
            self.table_relation = {}
            for rel in config['relations']:
                self.table_relation[rel['table_A']] = collections.defaultdict(list)
                self.table_relation[rel['table_B']] = collections.defaultdict(list)
            for rel in config['relations']:
                ta = rel['table_A']
                tb = rel['table_B']
                key = rel['key']
                for v in key:
                    self.table_relation[ta][v].append(tb)
                    self.table_relation[tb][v].append(ta)

        def dfs(graph, name, col, pset):
            if name in self.dataCovert.keys() and col in self.dataCovert[name].keys():
                return name

            if name in pset:
                return None

            pset.add(name)
            for table_name in graph[name][col]:
                relation_table_name = dfs(graph, table_name, col, pset)
                if relation_table_name:
                    try:
                        self.dataCovert[name][col] = self.dataCovert[relation_table_name][col]
                    except Exception:
                        self.dataCovert[name] = {}
                        self.dataCovert[name][col] = self.dataCovert[relation_table_name][col]
                    return relation_table_name

            return None

        for name, data in Xs.items():
            if not is_train and name != MAIN_TABLE_NAME:
                continue
            if is_train:
                if name not in self.dataCovert.keys():
                    self.dataCovert[name] = {}

            for col in data.columns:
                if name in self.table_relation.keys() and col in self.table_relation[name].keys():
                    pset = set()
                    data_covert = dfs(self.table_relation, name, col, pset)
                    if data_covert:
                        self.dataCovert[name][col].transform(data)
                    else:
                        self.dataCovert[name][col] = DataConvert(col)
                        self.dataCovert[name][col].fit_transform(data)
                elif is_train:
                    self.dataCovert[name][col] = DataConvert(col)
                    self.dataCovert[name][col].fit_transform(data)
                else:
                    self.dataCovert[name][col].transform(data)

    @timeit
    def get_sampe_size(self, Xs):
        total_time = self.config["time_budget"]
        if total_time >= 1200:
            return 50000
        if total_time >= 300:
            return 25000
        if total_time >= 200:
            print("get_sampe_size function less time sample_size = 20000")
            return 20000
        # aX + bn = time / 2
        # 根据上式计算采样数量，留出一半时间用于模型融合。a和b取值为经验值，单位：s/1W条数据
        # a 代表取样数据每万条特征工程+特征筛选所需时间
        # b 代表全量数据每万条处理所需时间
        a = 25
        b = 3
        n = Xs[MAIN_TABLE_NAME].shape[0] * 1.0 / 10000
        sample_size = int((total_time / 2 - b * n) / a * 10000)
        if sample_size <= 0:
            print("get_sampe_size function error sample_size <= 0")
            sample_size = 1000
        return sample_size

    @timeit
    def sampleXs(self, Xs, y):
        main_sample_size = self.get_sampe_size(Xs)
        tmp_Xs = {}
        tmp_Xs[MAIN_TABLE_NAME], tmp_y = down_sample_for_feature_iteration(Xs[MAIN_TABLE_NAME], y, sample_size=main_sample_size)
        log(f"{MAIN_TABLE_NAME} table sample size: {len(tmp_y)}")
        target_len= max(len(tmp_y), main_sample_size)
        max_time = tmp_Xs[MAIN_TABLE_NAME][self.config['time_col']].max()

        for name, data in Xs.items():
            if name == MAIN_TABLE_NAME:
                continue
            # 这里可能造成偏差，如果从表的行数小于主表，取主表采样的数目，可能会造成特征分布与全量数据上造出来的不同【再考虑】
            #sample_size = max(int(len(data) / len(Xs[MAIN_TABLE_NAME]) * len(tmp_Xs[MAIN_TABLE_NAME])), len(tmp_Xs[MAIN_TABLE_NAME]))
            if self.config['time_col'] in data:
                sample_size = max(int(len(data[data[self.config['time_col']] <= max_time]) / len(Xs[MAIN_TABLE_NAME]) * target_len), target_len)
                tmp_Xs[name] = data_sample(data[data[self.config['time_col']] <= max_time], sample_size)
            else:
                sample_size = max(int(len(data) / len(Xs[MAIN_TABLE_NAME]) * target_len), target_len)
                tmp_Xs[name] = data_sample(data, sample_size)
            log(f"{name} table sample size: {len(tmp_Xs[name])}")

        return tmp_Xs, tmp_y

    @timeit
    def sampleXsByKey(self, Xs, y):
        main_sample_size = self.get_sampe_size(Xs)
        tmp_Xs = {}
        tmp_Xs[MAIN_TABLE_NAME], tmp_y = down_sample_for_feature_iteration(Xs[MAIN_TABLE_NAME], y,
                                                                           sample_size=main_sample_size)
        log(f"{MAIN_TABLE_NAME} table sample size: {len(tmp_y)}")
        target_len = max(len(tmp_y), main_sample_size)
        # max_time = tmp_Xs[MAIN_TABLE_NAME][self.config['time_col']].max()

        visited = {MAIN_TABLE_NAME}

        def dfs(table_name):
            for relation in self.config['relations']:
                related_table_name = None
                if relation['table_A'] == table_name:
                    related_table_name = relation['table_B']
                if relation['table_B'] == table_name:
                    related_table_name = relation['table_A']
                if related_table_name is None or related_table_name in visited:
                    continue

                if len(Xs[related_table_name]) < target_len:
                    tmp_Xs[related_table_name] = Xs[related_table_name].copy()
                else:
                    related_table = Xs[related_table_name]
                    key = relation['key']
                    tmp_data = tmp_Xs[table_name][key].drop_duplicates()
                    sample_data = pd.merge(related_table, tmp_data)

                    if len(sample_data) > target_len*CONSTANT.SAMPLE_RATE_UP:  #过大减少
                        print(f"log: I'm in up")
                        sample_data = data_sample(sample_data, int(target_len*CONSTANT.SAMPLE_RATE_UP))
                    elif len(sample_data) < target_len*SAMPLE_RATE_DOWN: #过小，可能造成频率编码等特征效果太差, 先不加， 后面再测试
                        print(f"log: I'm in down")
                        sample_data = data_sample(Xs[related_table_name], target_len)
                        #pass
                    tmp_Xs[related_table_name] = sample_data

                log(f"{related_table_name} table sample size: {len(tmp_Xs[related_table_name])}, origin size: {len(Xs[related_table_name])}")
                visited.add(related_table_name)
                dfs(related_table_name)

        dfs(MAIN_TABLE_NAME)
        return tmp_Xs, tmp_y

    @timeit
    def feature_iteration(self, tmp_Xs, tmp_y, max_iter=2):
        '''
        在小批量数据上对特征进行筛选
        '''
        # 分表特征
        feature_iter_Xs = FeatureIterationXs(self.config)
        tmp_X = feature_iter_Xs.feature_engineering(tmp_Xs, tmp_y)
        print('total feature Xs: ', tmp_X.shape)
        del tmp_Xs

        TimeManager.set_separate_table_time(feature_iter_Xs.get_estimated_time_for_all_data())

        # 合并之后做大表特征
        feature_iter = FeatureIteration(self.config, feature_iter_Xs.hyperparams)

        # 一阶特征一定做 zyp
        for i in range(3):
            feature_iter.feature_engineering(tmp_X, tmp_y)

        # -----------------zyp resource control ---------------------------------------
        while TimeManager.still_have_time(feature_iter) and MemoryManager.check_memory(tmp_X, tmp_y):
            print(f'iter count: {feature_iter.index}')

            # 2nd level feature engineering
            if not feature_iter.feature_engineering(tmp_X, tmp_y):
                break

            if not (TimeManager.still_have_time(feature_iter) and MemoryManager.check_memory(tmp_X, tmp_y)):
                print('*********** top k selection ***************')
                top_k = feature_iter.top_k
                top_start = len(feature_iter.free_cols)
                top_end=top_k
                while top_end-top_start > 2:
                    top_mid = int((top_start+top_end)//2)
                    print(f'top_mid: {top_mid}')
                    if TimeManager.still_have_time(feature_iter,top_mid) and \
                            MemoryManager.check_memory(tmp_X, tmp_y, feature_iter.get_top_k_drop_cols(top_mid)):
                        top_start = top_mid
                    else:
                        top_end = top_mid
                top_k = top_start
                print(f'*********** top k ={top_k} **************8')

                if top_k < len(feature_iter.free_cols):
                    top_k = len(feature_iter.free_cols)

                try:
                    feature_iter.top_k_modify_map(tmp_X,top_k)
                except Exception:
                    pass
                #print(feature_iter.get_estimated_time_for_all_data())
                break
        # ----------------------------------------------------------------------------------

        print('total feature X: ', tmp_X.shape)
        del tmp_X, tmp_y
        gc.collect()
        return feature_iter_Xs, feature_iter

    @timeit
    def data_clean(self, Xs):
        key_attrs = set()
        for relation in self.config['relations']:
            key_attrs |= set(relation['key'])
        attrs2drop = {name: set() for name in Xs.keys()}
        for name, data in Xs.items():
            # remove the attributes whose values never changed
            attrs2drop[name] |= set(data.columns[data.nunique() == 1])

            # remove the numerical attributes whose values are the same
            num_feats = [c for c in data if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
            corr_matrix = data[num_feats].corr()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column].abs() >= 1)]
            attrs2drop[name] |= set(to_drop)

            attrs2drop[name] = set(attrs2drop[name]) - set(key_attrs)
        return attrs2drop

    @timeit
    def clean_data(self, Xs, tmp_Xs = None, isTrain=True):
        if isTrain:
            self.attrs2drop = self.data_clean(tmp_Xs)
        for name in Xs.keys():
            if (name == MAIN_TABLE_NAME or isTrain) and len(self.attrs2drop[name]) > 0:
                log(f"drop {name} table origin features : {self.attrs2drop[name]}")
                Xs[name].drop(list(self.attrs2drop[name]), axis=1, inplace=True)
                if tmp_Xs is not None:
                    tmp_Xs[name].drop(list(self.attrs2drop[name]), axis=1, inplace=True)

    @timeit
    def time_info_analysis(self, X, time_col):
        if time_col not in X.columns:
            self.config['time_series_index'] = [sorted(list(range(len(X))))]
            #print(self.config['time_series_index'][0])
            return
        time_types = ['year', 'month', 'day', 'hour', 'minute', 'second']

        split_list = []
        type_col = None
        for i, type in enumerate(time_types):
            type_col = getattr(X[time_col].dt, type)
            split_list = type_col.unique()
            if len(split_list) > 5 or type == 'second' or i == 5:
                print("jhy define type", type, "unique size", len(split_list))
                break
            elif len(split_list) > 1:
                type_col = getattr(X[time_col].dt, type) * 10000 + getattr(X[time_col].dt, time_types[i + 1])
                split_list = type_col.unique()
                print("jhy define warning!!! type ", type, "next and it's unique size", len(split_list))
                break

        time_series_index = []
        for v in split_list:
            cur_index = type_col[type_col == v].index.tolist()
            cur_index = list(cur_index)
            cur_index = sorted(cur_index)
            #print(cur_index[0:min(100, len(cur_index))])
            time_series_index.append(cur_index)
        print(f"time series info: time type {type}, could split {len(time_series_index)} segments")

        self.config['time_series_index'] = time_series_index
        self.config['time_series_type'] = type

    @timeit
    def fit(self, Xs, y, time_remain):

        gc.collect()

        # ------------- zyp for time control ------------------------
        TimeManager.init_time_control(Xs, self.config['time_budget'], time_remain, time.time())
        # ----------------------------------------------------------

        # main table sort by time col
        timer = Timer()
        seed_everything()
        if self.config['time_col'] in Xs[MAIN_TABLE_NAME].columns:
            Xs[MAIN_TABLE_NAME].sort_values(self.config['time_col'], inplace=True)
            y = y[Xs[MAIN_TABLE_NAME].index]
            Xs[MAIN_TABLE_NAME].reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

        self.convertX(Xs, self.config)
        self.time_info_analysis(Xs[MAIN_TABLE_NAME], self.config['time_col'])
        print(f"before sample, X shape: {Xs[MAIN_TABLE_NAME].shape}")
        Xs[MAIN_TABLE_NAME], y = time_series_pre_data_sample(Xs[MAIN_TABLE_NAME], y, self.config['time_series_index'], max_sample_size=1500000)
        Xs[MAIN_TABLE_NAME].reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        # 副表采样
        tables_data_sample(Xs, max_sample_size=1000000)

        # 针对副表特征太多，会产生级联的情况，对多对多的聚合操作进行控制
        col_cnt = 0
        for name, data in Xs.items():
            col_cnt += data.shape[1]
            print(name, data.shape)
        if col_cnt > 70 or len(Xs) > 3:
            CONSTANT.TOO_MANY_COLS = True
        print(f"after sample, X shape: {Xs[MAIN_TABLE_NAME].shape}", len(Xs))

        # 对主表进行100万采样之后，主表的时间分布已经和之前不一样了，因此要重新进行时间列分析
        self.time_info_analysis(Xs[MAIN_TABLE_NAME], self.config['time_col'])

        start_time=time.time()
        try:
            tmp_Xs, tmp_y = self.sampleXsByKey(Xs, y)
        except Exception:
            tmp_Xs, tmp_y = self.sampleXs(Xs, y)

        self.clean_data(Xs, tmp_Xs)
        end_time=time.time()

        # ---------------- train preprocess time ------------------------
        TimeManager.train_preprocess_time=end_time-start_time
        print(f'train preprocess time: {TimeManager.train_preprocess_time}')
        # ---------------- train preprocess time ------------------------

        # ----------feature engineering and selection iteration, time and mem control ---------------
        # ----------on small sample data ---------------
        self.feature_iter_Xs, self.feature_iter = self.feature_iteration(tmp_Xs, tmp_y)
        # ----------------------------------------------------------------------

        # ------------- Get feature on all data ---------------------------------
        CONSTANT.TIME_CONTROL = False

        self.Xs = Xs
        self.y = y
        return

    @timeit
    def predict(self, X_test, time_remain):
        timer = Timer()

        # -------- trace mem ----------------------
        #tracemalloc.start(3)
        # -------- trace mem ----------------------

        gc.collect()

        # ----- set mem for feature resume -------
        MemoryManager.set_avl_sys_mem()
        # ----- set mem for feature resumme -------
        #print(self.Xs[CONSTANT.MAIN_TABLE_NAME]['t_01'].min(), self.Xs[CONSTANT.MAIN_TABLE_NAME]['t_01'].max())
        #print(X_test['t_01'].min(), X_test['t_01'].max())

        X_test.reset_index(drop=True, inplace=True)
        if self.config['time_col'] in X_test.columns:
            X_test.sort_values(self.config['time_col'], inplace=True)
        index = X_test.index
        X_test.reset_index(drop=True, inplace=True)
        index = np.argsort(index)

        #print(f'X_test preprocess memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        X_main = self.Xs[CONSTANT.MAIN_TABLE_NAME]  # train main

        self.Xs[CONSTANT.MAIN_TABLE_NAME] = X_test
        del X_test
        self.convertX(self.Xs, self.config, False)
        self.clean_data(self.Xs, isTrain=False)

        #print(f'X_test convert memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        train_len = X_main.shape[0]
        self.Xs[CONSTANT.MAIN_TABLE_NAME] = pd.concat([X_main, self.Xs[CONSTANT.MAIN_TABLE_NAME]], axis=0).reset_index(
            drop=True)

        #print(f'concate memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        del X_main
        gc.collect()

        #print(f'del memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        X = self.feature_iter_Xs.feature_resume(self.Xs, self.y, isTrain=True)  # 分表作一阶特征
        #print(f'Xs feature memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        Xs_name_list = [name for name in self.Xs.keys()]
        for name in Xs_name_list:
            del self.Xs[name]
        del self.Xs
        gc.collect()

        #print(f'del Xs memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        # zcm修改，删除MC特征
        self.prep_class.drop_mulcat_features(X)
        gc.collect()
        # end zcm修改，删除MC特征

        CONSTANT.TRAIN_LEN = train_len
        self.feature_iter.feature_resume(X, self.y, isTrain=True)  # 大表作所有特征

        #print(f'X feature memory trace (now, peak): {tracemalloc.get_traced_memory()}')
        print(f'X mem after resume: {X.memory_usage().sum()}')

        self.prep_class.drop_features(X)
        gc.collect()
        #print(f'drop memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        print(f'final X mem: {X.memory_usage().sum()}')

        train(X.iloc[0:train_len], self.y, self.config, timer)
        gc.collect()

        #print(f'train memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        X_test = X.iloc[train_len:].reset_index(drop=True)
        del X
        gc.collect()
        result = predict(X_test, self.config)
        #print(f'predict memory trace (now, peak): {tracemalloc.get_traced_memory()}')

        result = result[index]
        return pd.Series(result)

