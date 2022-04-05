import pathlib

TRANSLATED_TEXTS = pathlib.Path('texts/wikipedia')
SHUFFLED_TEXTS = pathlib.Path('texts/shuffled')

SRC_LANG = 'ru'
DST_LANG = 'en'

# prepare texts
TEXTS_PIECE_SIZE = 3  # num of sentences in one piece

# shuffle texts
MIN_PIECE_NUMS = 3
MAX_PIECE_NUMS = 7

EACH_CATEGORY_TEXTS_NUM = 100
SRC_TEXTS_TO_SHUFFLE = 200
