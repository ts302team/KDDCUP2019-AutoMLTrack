from collections import defaultdict, deque
from util import *
import gc
from resource_manager import *
import preprocess

NUM_OP = [np.std, np.mean]

def bfs(root_name, graph, tconfig):
    """ Just add depth info by bfs for later dfs """
    tconfig[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
    queue = deque([root_name])
    while queue:
        u_name = queue.popleft()
        for edge in graph[u_name]:
            v_name = edge['to']
            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)


@timeit
def join(u, v, u_name, v_name, key, type_):
    if type_.split("_")[2] == 'many':
        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key}
        # sum and mean the numerical values, count the category type values of all rows with the same key.
        v = v.groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(lambda a: f"{CONSTANT.TIME_PREFIX}({a[0]})[{a[1].upper()}]" \
                                     if a[0].startswith(CONSTANT.TIME_PREFIX) else f"{CONSTANT.NUMERICAL_PREFIX}({a[0]})[{a[1].upper()}]") #这里cat_int类型会变成num
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_|{v_name}|({a})")
    u = u.join(v, on=key, how='left') # simple join according to the key.
    return u


def process_cur_key_uv(tmp_u, u_begin, v_begin, i):
    if u_begin == -1 or v_begin == -1:
        return
    if u_begin-v_begin >= i-u_begin: # v is more than u
        u_num=i-u_begin
        # print(f'**zyp *** {cur_key_begin},{u_begin},{u_num}')
        tmp_u[u_begin:i, -1] = tmp_u[u_begin-u_num:u_begin, 1]
        tmp_u[u_begin:i, -2] = (tmp_u[u_begin:i, 2] - tmp_u[u_begin-u_num:u_begin, 2]) // 1000000
    else:  # u is more than v
        v_num = u_begin - v_begin
        #if v_num > 0: # comment this line to fill u without v with the nearest v before, but v's key is different
        tmp_u[u_begin:u_begin+v_num, -1] = tmp_u[v_begin:u_begin, 1]
        tmp_u[u_begin:u_begin+v_num, -2] = (tmp_u[u_begin:u_begin+v_num, 2] - tmp_u[v_begin:u_begin, 2]) // 1000000

        tmp_u[u_begin+v_num:i, -1] = tmp_u[u_begin-1, 1]
        tmp_u[u_begin+v_num:i, -2] = (tmp_u[u_begin+v_num:i, 2] - tmp_u[u_begin-1, 2]) // 1000000
        #if v_num > 0:
        #    tmp_u[u_begin+v_num:i, -1] = tmp_u[u_begin-1, 1]
        #    tmp_u[u_begin+v_num:i, -2] = tmp_u[u_begin+v_num:i, 2] - tmp_u[u_begin-1, 2]
        #else:
        #    tmp_u[u_begin+v_num:i, -1] = tmp_u[u_begin-1, -1]
            #tmp_u[u_begin+v_num:i, -2] = tmp_u[u_begin+v_num:i, 2] - tmp_u[u_begin-1, 2]


@timeit
def temporal_join_opt1(u, v, u_name, v_name, key, time_col):
    # return u
    timer = Timer()

    if not isinstance(key, list):
        key = [key]

    tmp_v = v.copy(deep=True)
    tmp_u = u[[time_col] + key]
    tmp_u[time_col] = tmp_u[time_col].values.astype('int64')
    tmp_v[time_col] = tmp_v[time_col].values.astype('int64')

    if len(key) == 1:
        key = key[0]
    else:
        new_key = 'key_list'
        tmp_u[new_key] = tmp_u[key].astype(str).sum(axis=1)
        tmp_v[new_key] = tmp_v[key].astype(str).sum(axis=1)
        tmp_u = tmp_u.drop(key, axis=1)
        tmp_v = tmp_v.drop(key, axis=1)
        key = new_key

    timer.check("select")

    tmp_u = pd.concat([tmp_u, tmp_v[[time_col, key]]], keys=['u', 'v'], sort=False)
    timer.check("concat")

    times_col_name = f'{CONSTANT.CAT_TIME_PREFIX}{key}_times'
    times_diff_col_name = f'{CONSTANT.CAT_TIME_PREFIX}{key}_times_diff'

    tmp_u.sort_values([key, time_col], inplace=True)
    tmp_u.reset_index(inplace=True)
    tmp_u[times_col_name] = 1 #/n_u
    tmp_u[times_diff_col_name] = 0
    tmp_u['v_row'] = len(tmp_v)  # the initial value of v_row could be used to fill u without v
    columns = tmp_u.columns
    timer.check("sort")

    tmp_u = tmp_u.to_numpy()
    gc.collect()

    times = 0
    cur_key = tmp_u[0, 3]
    #cur_key_begin = 0
    u_begin = -1
    v_begin = -1
    if tmp_u[0, 0] == 'v':
        v_begin = 0
    if tmp_u[0, 0] == 'u':
        u_begin = 0

    for i in range(len(tmp_u)):
        if cur_key != tmp_u[i, 3]:
            # v to u
            process_cur_key_uv(tmp_u, u_begin, v_begin, i)
            #cur_key_begin = i
            u_begin = -1
            v_begin = -1
            if tmp_u[i, 0] == 'v':
                v_begin = i
            if tmp_u[i, 0] == 'u':
                u_begin = i
            times = 0
            cur_key = tmp_u[i, 3]

        if tmp_u[i, 0] == 'v':
            times = 0
            if u_begin != -1:
                process_cur_key_uv(tmp_u, u_begin, v_begin, i)
                v_begin = i
                u_begin = -1

        if tmp_u[i, 0] == 'u':
            if u_begin == -1:
                u_begin = i
            times = times + 1
            tmp_u[i, -3] = times  # /n_u

    # process the last key
    process_cur_key_uv(tmp_u, u_begin, v_begin, len(tmp_u))

    tmp_u = pd.DataFrame(tmp_u)
    tmp_u.columns = columns
    tmp_u.set_index('level_0', inplace=True)
    timer.check("update_row")

    #if tmp_u.empty:
    #    log("empty tmp_u, return u")
    #    return u

    tmp_u = tmp_u.loc['u'].reset_index(drop=True)
    tmp_u.drop([key, time_col], axis=1, inplace=True)

    tmp_v = tmp_v.drop([key, time_col], axis=1)

    # ---- if we want to use fillna to fill those u without v, use the following codes -----
    nan_row_df = pd.DataFrame([[np.nan] * len(tmp_v.columns)], columns=tmp_v.columns)
    preprocess.PreprocessClass().fillna(nan_row_df, only_one_row=True)
    tmp_v = tmp_v.append(nan_row_df, ignore_index=True)
    timer.check("fill na")
    #print(f'**zyp fillna a row {tmp_v.iloc[-1]}')
    # ---------------------------------------------------------------------------

    tmp_v = tmp_v.loc[tmp_u['v_row']].reset_index(drop=True)
    tmp_u = pd.concat([tmp_u, tmp_v], axis=1, sort=False)
    tmp_u = tmp_u.set_index('level_1').drop('v_row', axis=1)
    timer.check("uv concate")

    tmp_u[times_col_name] = pd.to_numeric(tmp_u[times_col_name], downcast='signed')
    #tmp_u[times_col_name] = tmp_u[times_col_name].astype("uint8")
    print("time diff zcm check : ", tmp_u[times_diff_col_name].max())
    tmp_u[times_diff_col_name] = pd.to_numeric(tmp_u[times_diff_col_name], downcast='signed')
    #tmp_u[times_diff_col_name] = tmp_u[times_diff_col_name].astype("int32")
    tmp_u.columns = tmp_u.columns.map(lambda a: f"{a.split('_', 1)[0]}_|{v_name}|({a})")

    tmp_u = pd.concat([u, tmp_u], axis=1, sort=False)
    timer.check("final concat")
    return tmp_u

def dfs(u_name, config, tables, graph):
    u = tables[u_name]
    log(f"enter {u_name}")
    for edge in graph[u_name]:
        v_name = edge['to']
        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)

        key = edge['key']
        type_ = edge['type']

        log(f"join {u_name} <--{type_}--t {v_name}")
        print(u_name, u.shape, v_name, v.shape)

        if config['time_col'] not in u and config['time_col'] in v:

            if CONSTANT.TIME_CONTROL:
                start_time=time.time()
                u = join(u, v, u_name, v_name, key, type_)
                end_time=time.time()
                TimeManager.add_ordinary_join_item(len(u.index),len(v.index),CONSTANT.TABLE_LENGTHS[u_name],
                                                   CONSTANT.TABLE_LENGTHS[v_name],end_time-start_time)
            # ----------------------------------------------------------------
            else:
                u = join(u, v, u_name, v_name, key, type_)
        elif config['time_col'] in u and config['time_col'] in v:
            # ----------------- time control --------------------------------
            if CONSTANT.TIME_CONTROL:
                start_time=time.time()
                u = temporal_join_opt1(u, v, u_name, v_name, key, config['time_col'])
                #u = join(u, v, u_name, v_name, key, type_)
                end_time=time.time()
                TimeManager.add_temporal_join_item(len(u.index),len(v.index),CONSTANT.TABLE_LENGTHS[u_name],
                                                   CONSTANT.TABLE_LENGTHS[v_name],end_time-start_time)
            # --------------------------------------------------------------------
            else:
                u = temporal_join_opt1(u, v, u_name, v_name, key, config['time_col'])
                #u = join(u, v, u_name, v_name, key, type_)
        else:
            # ----------------- time control --------------------------------
            if CONSTANT.TIME_CONTROL:
                start_time=time.time()
                u = join(u, v, u_name, v_name, key, type_)
                end_time=time.time()
                TimeManager.add_ordinary_join_item(len(u.index),len(v.index),CONSTANT.TABLE_LENGTHS[u_name],
                                                   CONSTANT.TABLE_LENGTHS[v_name],end_time-start_time)
            # ----------------------------------------------------------------
            else:
                u = join(u, v, u_name, v_name, key, type_)
        del v
    log(f"leave {u_name}")
    return u


@timeit
def merge_table(tables, config):
    graph = defaultdict(list)
    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']
        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })
        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })
    bfs(CONSTANT.MAIN_TABLE_NAME, graph, config['tables'])
    return dfs(CONSTANT.MAIN_TABLE_NAME, config, tables, graph)

