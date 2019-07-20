
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