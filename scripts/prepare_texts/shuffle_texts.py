import collections
import hashlib
import nltk
import pathlib
import random

from progress import bar

from scripts import consts


_COMBINED_TEXTS_NUM = 5
_TOTAL_TEXTS_NUM = 10000


def main():
    src_lang_texts = traverse_texts(consts.TRANSLATED_TEXTS.joinpath(consts.SRC_LANG))
    dst_lang_texts = traverse_texts(consts.TRANSLATED_TEXTS.joinpath(consts.DST_LANG))
    shuffled, originality = shuffle_texts(src_lang_texts)
    make_texts(shuffled, src_lang_texts, consts.SRC_LANG)
    make_texts(shuffled, dst_lang_texts, consts.DST_LANG)
    with open(consts.SHUFFLED_TEXTS.joinpath('originality'), 'w') as out_file:
        out_file.write(str(dict(originality)))
    


def traverse_texts(root_dir: pathlib.Path) -> str:
    res = {}
    for txt_file in root_dir.glob('*'):
        with open(txt_file) as inp_file:
            res[txt_file.stem] = inp_file.read()
    return res


def shuffle_texts(src_texts):
    texts_originalities = collections.defaultdict(dict)
    result_texts = {}
    texts_lenghts = get_texts_lengths(src_texts)
    progress_bar = bar.IncrementalBar(
        'Shuffled texts', 
        max=_TOTAL_TEXTS_NUM
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
        

def make_texts(texts_hashes, texts, lang):
    progress_bar = bar.IncrementalBar(
        f'Generated {lang} texts', 
        max=len(texts_hashes)
    )
    for file_name, text_hashes in texts_hashes.items():
        dir_path = consts.SHUFFLED_TEXTS.joinpath(lang)
        dir_path.mkdir(
            parents=True, exist_ok=True
        )
        text = ' '.join(
            [texts[text_hash] for text_hash in text_hashes]
        )
        with open(dir_path.joinpath(file_name), 'w') as out_file:
            out_file.write(text)
        progress_bar.next()
    progress_bar.finish()


def get_texts_lengths(src_lang_texts):
    texts_lenghts = {}
    for name, text in src_lang_texts.items():
        words_num = len(nltk.tokenize.word_tokenize(text))
        texts_lenghts[name] = words_num
    return texts_lenghts


if __name__ == '__main__':
    main()
