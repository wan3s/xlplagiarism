import collections
import hashlib
import nltk
import pathlib
import random
import json

from progress import bar

from scripts import consts


def main():
    src_lang_texts = traverse_texts(consts.TRANSLATED_TEXTS.joinpath(consts.SRC_LANG))
    dst_lang_texts = traverse_texts(consts.TRANSLATED_TEXTS.joinpath(consts.DST_LANG))
    shuffled, originality = shuffle_texts(
        {name: src_lang_texts[name] for name in list(src_lang_texts.keys())[:consts.SRC_TEXTS_TO_SHUFFLE]}
    )
    make_texts(shuffled, src_lang_texts, consts.SRC_LANG)
    make_texts(shuffled, dst_lang_texts, consts.DST_LANG)
    with open(consts.SHUFFLED_TEXTS.joinpath('originality'), 'w') as out_file:
        out_file.write(json.dumps(originality))
    


def traverse_texts(root_dir: pathlib.Path) -> str:
    res = {}
    for txt_file in root_dir.glob('*'):
        with open(txt_file) as inp_file:
            res[txt_file.stem] = inp_file.read()
    return res


def shuffle_texts(src_texts):
    subdir_index = 0
    subdirs_num = len(consts.SHUFFLED_TEXTS_SUBDIRS)
    texts_originalities = collections.defaultdict(lambda: collections.defaultdict(dict))
    result_texts = collections.defaultdict(dict)
    texts_lenghts = get_texts_lengths(src_texts)
    total_texts_num = (consts.MAX_PIECE_NUMS - consts.MIN_PIECE_NUMS + 1) * consts.EACH_CATEGORY_TEXTS_NUM
    progress_bar = bar.IncrementalBar(
        'Shuffled texts', 
        max=total_texts_num
    )
    for combined_texts_num in range(consts.MIN_PIECE_NUMS, consts.MAX_PIECE_NUMS + 1):
        for _ in range(consts.EACH_CATEGORY_TEXTS_NUM):
            new_text = []
            while len(new_text) < combined_texts_num:
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
            dataset_name = consts.SHUFFLED_TEXTS_SUBDIRS[subdir_index]
            for text_hash in new_text:
                texts_originalities[dataset_name][new_text_name][text_hash] = (
                    texts_lenghts[text_hash] / total_len
                )
            result_texts[dataset_name][new_text_name] = new_text
            subdir_index = (subdir_index + 1) % subdirs_num
            progress_bar.next()
    progress_bar.finish()
    return result_texts, texts_originalities
        

def make_texts(texts_hashes_by_subdirs, texts, lang):
    for subdir, texts_hashes in texts_hashes_by_subdirs.items():
        progress_bar = bar.IncrementalBar(
            f'Generated {lang} texts to {subdir} dataset', 
            max=len(texts_hashes)
        )
        for file_name, text_hashes in texts_hashes.items():
            dir_path = consts.SHUFFLED_TEXTS.joinpath(f'{subdir}/{lang}')
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
