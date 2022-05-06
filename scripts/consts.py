import pathlib

TRANSLATED_TEXTS = pathlib.Path('xlplagiarism/texts/wikipedia')
SHUFFLED_TEXTS = pathlib.Path('xlplagiarism/texts/shuffled')

SRC_LANG = 'ru'
DST_LANG = 'en'

# prepare texts
TEXTS_PIECE_SIZE = 3  # num of sentences in one piece

# shuffle texts
MIN_PIECE_NUMS = 3
MAX_PIECE_NUMS = 7

VALIDATION_DATASET = 'validation'
TEST_DATASET = 'test'

SHUFFLED_TEXTS_SUBDIRS = [
    VALIDATION_DATASET,
    TEST_DATASET,
]

# labse
LABSE_SIM_THRESHOLD = 0.7

EACH_CATEGORY_TEXTS_NUM = 100
SRC_TEXTS_TO_SHUFFLE = 200
