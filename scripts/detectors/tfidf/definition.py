import collections
import deep_translator
import math
import nltk
import typing as tp
from progress import bar

from scripts.detectors import common
from scripts import consts


class TfIdfDetector(common.BaseDetector):
    def __init__(self, *args, translate=None, **kwargs) -> None:
        print('TfIdfDetector initializing ...')
        super().__init__(*args, **kwargs)
        self._idf = count_idf(root_dir=consts.TRANSLATED_TEXTS)
        self._translator = deep_translator.GoogleTranslator(
            source=consts.SRC_LANG, 
            target=consts.DST_LANG
        )
        if translate is not None:
            self._translate = translate
        else:
            self._translate = True

    def count_similiarity(self, src_lang_text, dst_lang_text):
        if self._translate:
            translated_text = common.translate_text(self._translator, src_lang_text)
            src_tf_ifd = self.count_tf_idf(translated_text)
        else:
            src_tf_ifd = self.count_tf_idf(src_lang_text)
        dst_tf_idf = self.count_tf_idf(dst_lang_text)
        words_union = list(set().union(src_tf_ifd, dst_tf_idf))
        v1, v2 = _get_vect_by(words_union, src_tf_ifd), _get_vect_by(words_union, dst_tf_idf)
        return cos_between_vectors(v1, v2)

    def count_tf_idf(self, text):
        tf = count_tf(text)
        return {
            word: word_tf * self._idf.get(word, self._idf['__default__']) 
            for word, word_tf in tf.items()
        }


def _get_vect_by(words_union, tf_idf):
    return [
        tf_idf.get(word, 0) for word in words_union
    ]


def _split_text(text: str) -> tp.List[str]:
    return [word.lower() for word in nltk.tokenize.word_tokenize(text) if len(word) > 1]


def _vect_len(vect):
    return math.sqrt(
        sum(
            v * v for v in vect
        )
    )

def count_idf(root_dir):
    idf = collections.defaultdict(int)
    files_paths = [path for path in root_dir.glob(f'{consts.DST_LANG}/*')]
    n = len(files_paths)
    progress_bar = bar.IncrementalBar('Counting idf', max=n)
    for src_lang_file_path in files_paths:
        with open(src_lang_file_path) as inp_file:
            text = inp_file.read()
        words = set(_split_text(text))
        for word in words:
            idf[word] += 1
        progress_bar.next()
    progress_bar.finish()
    for word, word_idf in idf.items():
        idf[word] = math.log(n / float(word_idf))
    idf['__default__'] = math.log(n)
    return idf


def count_tf(text):
    counter = collections.Counter(_split_text(text))
    total_len = len(text)
    return {word: num / total_len for word, num in counter.items()}


def cos_between_vectors(v1, v2):
    vect_sum = sum([
        v1 * v2 for v1, v2 in zip(v1, v2)
    ])
    return vect_sum / (_vect_len(v1) * _vect_len(v2))
