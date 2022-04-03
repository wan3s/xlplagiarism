import pathlib

from detectors.labse import definition as labse_definition
from detectors.tfidf import definition as tfifd_definition
from prepare_texts import common

SHUFFLED_TEXTS = pathlib.Path('../shuffled')

def main():
    labse_detector = labse_definition.LabseDetector()
    # tfidf_detector = tfifd_definition.TfIdfDetector()
    for src_lang_file_path in common.TRANSLATED_TEXTS.glob(f'{common.SRC_LANG}/*'):
        with open(src_lang_file_path) as inp_file:
            src_lang_text = inp_file.read()
        filename = src_lang_file_path.stem
        dst_lang_file_path = common.TRANSLATED_TEXTS.joinpath(
            f'{common.DST_LANG}/{filename}'
        )
        with open(dst_lang_file_path) as inp_file:
            dst_lang_text = inp_file.read()
        print(
            f'{filename}: {labse_detector.count_similiarity(src_lang_text, dst_lang_text)}'
            # f'{filename}: {tfidf_detector.count_similiarity(src_lang_text, dst_lang_text)}'
        )


if __name__ == '__main__':
    main()
