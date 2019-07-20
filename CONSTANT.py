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

NUMERICAL_TYPE = "num"
NUMERICAL_PREFIX = "n_"

CATEGORY_TYPE = "cat"
CATEGORY_PREFIX = "c_"

CATEGORY_HIGH_TYPE = "cat_hig"
CATEGORY_HIGH_PREFIX = "ch_"

TIME_TYPE = "time"
TIME_PREFIX = "t_"

MULTI_CAT_TYPE = "multi-cat"
MULTI_CAT_PREFIX = "m_"
MULTI_CAT_DELIMITER = ","

MULTI_CAT_FRE_TYPE = "multi-cat-fre"
MULTI_CAT_FRE_PREFIX = "cimf_"

CAT_FRE_TYPE = "cat-int-fre"
CAT_FRE_PREFIX = "cif_"

CAT_TIME_TYPE = "time-int"
CAT_TIME_PREFIX = "cit_"

CAT_UNI_TYPE = "uni-int"
CAT_UNI_PREFIX = "ciu_"

CAT_INT_TYPE = 'cat-int'
CAT_INT_PREFIX = "ci"

MAIN_TABLE_NAME = "main"
MAIN_TABLE_TEST_NAME = "main_test"
TABLE_PREFIX = "table_"

LABEL = "label"

HASH_MAX = 200

# below are not contants, but put them here globally for convenience.
TIME_CONTROL = True

TABLE_LENGTHS={}

NAME_MAP={}
REVERSE_NAME_MAP = {}
CAT_UNI_INDEX=0
CAT_FRE_INDEX=0
NUM_INDEX=0
CAT_TIME_INDEX=0
MULTI_CAT_FRE_INDEX = 0
RENAME_SIGN="*"
CATEGORY_HIGH_INDEX=0


fillna_map = {}
ctr_has_done = set()
second_cat2cat_has_done = set()
count_encoder_has_done = set()
fre_has_done = set()
unique_cat_has_done = {}
time_cat_has_done = set()
cat_time_diff_has_done = set()

MAX_CAT_NUM = 1000000
MAX_UNIQUE_ENCODE = 200
MAX_CAT_CAT_ENCODE = 200
MAX_CAT_NUM_ENCODE = 50
cat_params = {}

MIN_FEAT_IMPORTANT = 0
MIN_CAT_FEAT_IMPORTANT = 50

MAX_SECOND_RATE = 2
MIN_NUM_SECOND_FEATURES = 250

MAX_MODEL_SIZE = 10

TRAIN_LEN = 0

SAMPLE_RATE_UP = 3
SAMPLE_RATE_DOWN = 0.1

TOO_MANY_COLS = False

MAX_FEATURE_ITERATIONS = 10
