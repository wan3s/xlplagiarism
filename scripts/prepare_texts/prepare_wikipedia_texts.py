import hashlib
import nltk
import pathlib
import typing as tp

import deep_translator

from progress import bar

from scripts import consts


_RAW_TEXTS_ROOT = consts.TRANSLATED_TEXTS.joinpath('_raw')

def traverse_raw_texts(root_dir: pathlib.Path) -> str:
    all_texts = ''
    for txt_file in root_dir.glob('**/*.txt'):
        with open(txt_file) as inp_file:
            all_texts += f'{inp_file.read()}\n'
    return all_texts


def split_raw_texts(raw_texts: str) -> tp.List[str]:
    result = []
    splited_sentences = nltk.sent_tokenize(raw_texts)
    while (splited_sentences):
        result.append(
            ''.join(splited_sentences[:consts.TEXTS_PIECE_SIZE])
        )
        splited_sentences = splited_sentences[consts.TEXTS_PIECE_SIZE:]
    return result


def translate_texts(
    texts: tp.List[str], 
    dst_lang: str = consts.DST_LANG
) -> tp.Dict[str, tp.Dict[str, str]]:
    result = {}
    translator = deep_translator.GoogleTranslator(source=consts.SRC_LANG, target=dst_lang)
    progress_bar = bar.IncrementalBar('Translated texts', max=len(texts))
    for text in texts:
        file_name = _get_file_name_by_text(text)
        translated_text = translator.translate(text)
        result[file_name] = {
            consts.SRC_LANG: text,
            dst_lang: translated_text
        }
        progress_bar.next()
    progress_bar.finish()
    return result


def save_translated_texts(translated_texts: tp.Dict[str, tp.Dict[str, str]]):
    progress_bar = bar.IncrementalBar(
        'Saved texts', 
        max=len(translated_texts)
    )
    for file_name, texts in translated_texts.items():
        for lang, text in texts.items():
            dir_path = consts.TRANSLATED_TEXTS.joinpath(lang)
            dir_path.mkdir(
                parents=True, exist_ok=True
            )
            with open(dir_path.joinpath(file_name), 'w') as out_file:
                out_file.write(text)
        progress_bar.next()
    progress_bar.finish()


def _get_file_name_by_text(text):
    hash_object = hashlib.md5(text.encode())
    return hash_object.hexdigest()


def main():
    raw_texts = traverse_raw_texts(_RAW_TEXTS_ROOT)
    splitted_texts = split_raw_texts(raw_texts)
    print(
        f'Gotten {len(splitted_texts)} texts '
        f'contained <= {consts.TEXTS_PIECE_SIZE} sentences'
    )
    translated_texts = translate_texts(splitted_texts)
    print(
        f'Num of translated texts: {len(translated_texts)}'
    )
    save_translated_texts(translated_texts)
    print('Completed!')


if __name__ == '__main__':
    main()
