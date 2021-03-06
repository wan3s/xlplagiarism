import deep_translator
import hashlib
import nltk

from scripts.detectors import common
from scripts import consts

_SHINGLE_LEN = 3

class ShinglesDetector(common.BaseDetector):
    def __init__(self, *args, translate=None, **kwargs) -> None:
        print('ShinglesDetector initializing ...')
        super().__init__(*args, **kwargs)
        self._translator = deep_translator.GoogleTranslator(
            source=consts.SRC_LANG, 
            target=consts.DST_LANG
        )
        if translate is not None:
            self._translate = translate
        else:
            self._translate = True

    def count_similiarity(self, src_lang_text: str, dst_lang_text: str):
        if self._translate:
            translated_text =  common.translate_text(self._translator, src_lang_text)
            src_hashes = _shingle_text(translated_text)
        else:
            src_hashes = _shingle_text(src_lang_text)
        dst_hashes = _shingle_text(dst_lang_text)
        return len(src_hashes.intersection(dst_hashes)) / len(src_hashes)


def _hash_list(lst):
    m = hashlib.md5()
    for s in lst:
        m.update(s.encode())
    return m.hexdigest()


def _shingle_text(text):
    words_hashes = [
        hashlib.md5(word.lower().encode('utf-8')).hexdigest()
        for word in nltk.tokenize.word_tokenize(text)
        if len(word) > 1
    ]

    words_num = len(words_hashes)
    if words_num < _SHINGLE_LEN:
        raise RuntimeError(f'Too short text {text}')

    set_of_shingles = set()

    for i in range(words_num + 1 - _SHINGLE_LEN):
        shingle = _hash_list(words_hashes[i:i + _SHINGLE_LEN])
        set_of_shingles.add(shingle)

    return set_of_shingles
