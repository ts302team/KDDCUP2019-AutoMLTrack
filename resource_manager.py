import time, CONSTANT
import os

SEPARATE_TABLE_TIME_RATIO = 1
MERGED_TABLE_TIME_RATIO = 2
PREDICT_FEATURE_TIME_RATIO = 2
TEMPORAL_JOIN_TIME_RATIO = 1
ORDINARY_JOIN_TIME_RATIO = 1
PREDICT_TIME_RATIO = 0.1
TRAIN_MODEL_RATIO = 0.3
MEM_PEAK_RATIO = 3

USE_PSUTIL = False

if USE_PSUTIL:
    import psutil


class TimeManager:
    start_time = 0
    time_budget = 0  # Time budget in info.json
    time_reserved = 0  # Time budget - reserved time
    time_remain=0

    merge_table_time = 0
    train_model_time = 0
    predict_feature_time = 0
    predict_model_time = 0

    merge_sample_table_time = 0

    temporal_join_list=[]
    ordinary_join_list=[]

    # unit: second
    separate_table_time_estimate_for_all_data = 0
    merged_table_time_estimate_for_all_data = 0

    #merge_table_time_added = False

    # ------------- accurate time ---------------------
    train_preprocess_time = 0
    train_feature_time = 0
    # ------------- accurate time ---------------------

    def __init__(self):
        TimeManager.separate_table_time_estimate_for_all_data = 0
        TimeManager.merged_table_time_estimate_for_all_data = 0

    @staticmethod
    def get_time_ratio(total_rows):
        pass


    @staticmethod
    def set_separate_table_time(seconds):
        TimeManager.separate_table_time_estimate_for_all_data = seconds

    #@staticmethod
    #def add_merged_table_time(seconds):
    #    TimeManager.merged_table_time_estimate_for_all_data += seconds

    @staticmethod
    def get_separate_table_time_estimate():
        return TimeManager.separate_table_time_estimate_for_all_data

    @staticmethod
    def get_merged_table_time_estimate():
        return TimeManager.merged_table_time_estimate_for_all_data

    @staticmethod
    def reset_time():
        TimeManager.merged_table_time_estimate_for_all_data=0
        TimeManager.separate_table_time_estimate_for_all_data=0

    @staticmethod
    def add_temporal_join_item(sample_u_rows,sample_v_rows,all_u_rows,all_v_rows, cost_time):
        TimeManager.temporal_join_list.append([sample_u_rows,sample_v_rows,all_u_rows,all_v_rows,cost_time])

    @staticmethod
    def add_ordinary_join_item(sample_u_rows,sample_v_rows,all_u_rows,all_v_rows,cost_time):
        TimeManager.ordinary_join_list.append([sample_u_rows,sample_v_rows,all_u_rows,all_v_rows,cost_time])

    @staticmethod
    def init_time_control(Xs, time_budget,time_remain,start_time):
        CONSTANT.TIME_CONTROL = True
        TimeManager.time_budget = time_budget
        TimeManager.time_remain = time_remain
        TimeManager.cal_reserved_time()
        TimeManager.start_time = start_time
        TimeManager.merge_table_time_added = False
        TimeManager.ordinary_join_list=[]
        TimeManager.temporal_join_list=[]
        TimeManager.merge_table_time = 0

        # - ------------------------------------------------------------
        # write these time calculation here temporarily.
        # Some cannot be calculated here.
        #TimeManager.merge_table_time = TimeManager.get_merge_table_time()
        #TimeManager.train_model_time = TimeManager.get_train_model_time(Xs)
        #TimeManager.predict_feature_time = TimeManager.get_predict_feature_time(Xs)
        #TimeManager.predict_model_time = TimeManager.get_predict_model_time(Xs)
        # -------------------------------------------------------------

        CONSTANT.TABLE_LENGTHS={}
        for name, data in Xs.items():
            CONSTANT.TABLE_LENGTHS[name] = len(data.index)
        TimeManager.reset_time()

    @staticmethod
    def cal_reserved_time():
        TimeManager.time_reserved = TimeManager.time_budget*0.05

    @staticmethod
    def get_time_left():
        return TimeManager.time_remain-TimeManager.time_reserved-(time.time()-TimeManager.start_time)

    @staticmethod
    def simple_check_time():
        ABSOLUTE_TIME = 300 # seconds
        TIME_RATIO = 0.3
        real_time_bound = min(TimeManager.time_budget * TIME_RATIO, ABSOLUTE_TIME)
        #print(TimeManager.get_time_left(),real_time_bound)
        return TimeManager.get_time_left() > real_time_bound

    @staticmethod
    def still_have_time(feature_iter,top_k=-1):
        try:
            cost_time_estimate = feature_iter.get_estimated_time_for_all_data(top_k)
        except Exception:
            cost_time_estimate = 0

        #TimeManager.predict_feature_time = cost_time_estimate / PREDICT_FEATURE_TIME_RATIO
        print(f"merge table time: {TimeManager.get_merge_table_time()}")
        print(f"seperate table time: {TimeManager.separate_table_time_estimate_for_all_data}")
        print(f"merged table time: {cost_time_estimate}")
        print(f"predict time: {TimeManager.get_predict_time()}")
        print(f"train model time: {TimeManager.get_train_model_time()}")
        #cost_time_estimate = cost_time_estimate + TimeManager.get_merge_table_time() + TimeManager.get_train_model_time() + TimeManager.get_predict_model_time() \
        #                     + TimeManager.predict_feature_time + TimeManager.separate_table_time_estimate_for_all_data
        cost_time_estimate = cost_time_estimate + TimeManager.get_merge_table_time() + TimeManager.get_train_model_time() \
                             + TimeManager.get_predict_time() + TimeManager.separate_table_time_estimate_for_all_data
        print(f'time left: {TimeManager.get_time_left()},time_cost_estimated: {cost_time_estimate}')

        ## RIGHT CODE
        return TimeManager.get_time_left() > cost_time_estimate
        ## JUST FOR DEBUG
        #return TimeManager.get_time_left() > cost_time_estimate+50

    @staticmethod
    def get_predict_time():
        return TimeManager.time_budget*PREDICT_TIME_RATIO

    @staticmethod
    def get_merge_table_time():
        if TimeManager.merge_table_time > 0:
            return TimeManager.merge_table_time
        total_time_estimate = 0
        for item in TimeManager.temporal_join_list:
            #total_time_estimate += (item[2]/item[0]*(item[3]/item[1])*item[4])
            total_time_estimate += (item[2]/item[0]*item[4])/TEMPORAL_JOIN_TIME_RATIO

        for item in TimeManager.ordinary_join_list:
            #total_time_estimate += (item[2]/item[0]*(item[3]/item[1])*item[4])
            total_time_estimate += (item[2]/item[0]*item[4])/ORDINARY_JOIN_TIME_RATIO

        TimeManager.merge_table_time = total_time_estimate
        print(f'merge table time esti: {total_time_estimate}')
        return total_time_estimate

    @staticmethod
    def get_train_model_time():
        return TimeManager.time_budget * TRAIN_MODEL_RATIO

    @staticmethod
    def get_predict_feature_time_by_real_time():
        return TimeManager.train_feature_time/2

    @staticmethod
    def get_predict_preprocess_time():
        return TimeManager.train_preprocess_time/2


class TimeControlObject:
    def __init__(self, all_rows, sample_rows):
        self.sample_duration = 0
        self.all_data_duration_estimate = {'extra': 0}
        self.all_data_rows = all_rows
        self.sample_rows = sample_rows

    def get_all_data_time_estimate(self,needed_type=None):
        total_time = 0
        if needed_type:
            try:
                total_time += self.all_data_duration_estimate[needed_type]
            except Exception:
                total_time += 0
        else:
            for t in self.all_data_duration_estimate.keys():
                total_time += self.all_data_duration_estimate[t]

        return total_time

    def remove_type_time(self,type):
        try:
            self.all_data_duration_estimate.pop(type)
        except:
            pass


    def set_col_time_estimate(self,type,duration):
        if CONSTANT.TIME_CONTROL:
            if self.sample_rows < 25000:
                self.sample_rows = min(25000, self.all_data_rows)//3
            self.all_data_duration_estimate[type] = duration*(self.all_data_rows/self.sample_rows)



class MemoryManager:
    reserved_mem = 0 #unit byte
    avl_sys_mem = 0
    @staticmethod
    def simple_check_mem(df_X,df_y,extra_sub_mem=0,extra_add_mem=0):
        #print(MemoryManager.avl_sys_mem, df_X.memory_usage().sum(), df_y.memory_usage(), extra_sub_mem)
        #print(MemoryManager.avl_sys_mem , (df_X.memory_usage().sum()+df_y.memory_usage() - extra_sub_mem + extra_add_mem) * 4)
        return MemoryManager.avl_sys_mem - MemoryManager.reserved_mem > \
               (df_X.memory_usage().sum() + df_y.memory_usage() - extra_sub_mem + extra_add_mem) * 4

    @staticmethod
    def set_avl_sys_mem():
        MemoryManager.avl_sys_mem = MemoryManager.get_avl_sys_mem()

    @staticmethod
    def get_avl_sys_mem():
        if USE_PSUTIL:
            return psutil.virtual_memory().available
        else:
            return int(os.popen('free -b').readlines()[2].split()[3])

    @staticmethod
    def check_memory(df_X, df_y, drop_cols=None, **kwargs):
        if drop_cols:
            df_X = df_X.drop(drop_cols,axis=1)
        ratio = CONSTANT.TABLE_LENGTHS[CONSTANT.MAIN_TABLE_NAME]/len(df_X.index)
        try:
            df_X_mem = df_X.memory_usage().sum()
        except:
            df_X_mem = df_X.memory_usage()
        df_y_mem=df_y.memory_usage()
        #tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        #avl_mem = psutil.virtual_memory().available-MemoryManager.reserved_mem+df_X_mem+df_y_mem
        avl_mem = MemoryManager.get_avl_sys_mem()-MemoryManager.reserved_mem+df_X_mem+df_y_mem
        mem_estimated = (df_X_mem+df_y_mem)*ratio
        for key, value in kwargs:
            if key == 'extra_mem':
                mem_estimated += value
            elif key == 'extra_dfs':
                mem_estimated += sum([df.memory_usage().sum() for df in value])
        print(f'avl_mem: {avl_mem}, esti_mem:{mem_estimated*MEM_PEAK_RATIO}')
        return avl_mem > mem_estimated*MEM_PEAK_RATIO

