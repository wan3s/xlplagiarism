import collections
import hashlib
import nltk
import pathlib
import random

from progress import bar

from . import common


_COMBINED_TEXTS_NUM = 5
_TOTAL_TEXTS_NUM = 10000


def main():
    texts = traverse_texts(common.TRANSLATED_TEXTS.joinpath(common.SRC_LANG))
    shuflled, originality = shuffle_texts(texts)


def traverse_texts(root_dir: pathlib.Path) -> str:
    res = {}
    for txt_file in root_dir:
        with open(txt_file) as inp_file:
            res[txt_file] = inp_file.read()
    return res


def shuffle_texts(src_texts):
    texts_originalities = collections.defaultdict(dict)
    result_texts = {}
    texts_lenghts = get_texts_lengths(src_texts)
    progress_bar = bar.IncrementalBar(
        'Shuffled texts', 
        max=len(_TOTAL_TEXTS_NUM)
    )
    while len(result_texts) < _TOTAL_TEXTS_NUM:
        new_text = []
        while len(new_text) < _COMBINED_TEXTS_NUM:
            cur_text = random.choice(list(src_texts.keys()))
            if cur_text not in new_text:
                new_text.append(cur_text)
        hash_object = hashlib.md5(
            str(new_text).encode()
        )
        new_text_name = hash_object.hexdigest()
        total_len = sum(
            [texts_lenghts[text_hash] for text_hash in new_text]
        )
        for text_hash in new_text:
            texts_originalities[new_text_name][text_hash] = (
                texts_lenghts[text_hash] / total_len
            )
        result_texts[new_text_name] = new_text
        progress_bar.next()
    progress_bar.finish()
    return result_texts, texts_originalities
        


def get_texts_lengths(src_lang_texts):
    texts_lenghts = {}
    for name, text in src_lang_texts.items():
        words_num = len(nltk.tokenize.word_tokenize(text))
        texts_lenghts[name] = words_num
    return texts_lenghts


if __name__ == '__main__':
    main()
