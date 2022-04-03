import pathlib

from detectors.labse import definition as labse_definition
from detectors.tfidf import definition as tfifd_definition
from detectors.shingles import definition as shingles_definition
from detectors import common as detectors_common
from prepare_texts import common

from progress import bar

import deep_translator

def mmain():
    labse_detector = labse_definition.LabseDetector()
    tfidf_detector = tfifd_definition.TfIdfDetector()
    shingles_detector = shingles_definition.ShinglesDetector()
    with open(common.SHUFFLED_TEXTS.joinpath('originality')) as inp_file:
        originality = eval(inp_file.read())
    print('Preparing completed!')
    src_texts_paths = [path for path in common.SHUFFLED_TEXTS.glob(f'{common.SRC_LANG}/*')][:10]
    # progress_bar = bar.IncrementalBar('Comparison', max=len(src_texts_paths))
    for src_lang_file_path in src_texts_paths:
        filename = src_lang_file_path.stem
        with open(src_lang_file_path) as inp_file:
            src_lang_text = inp_file.read()
        for dst_lang_file_path, acc in originality[filename]:
            with open(common.SHUFFLED_TEXTS.joinpath(f'{common.DST_LANG}/{dst_lang_file_path}')) as inp_file:
                dst_lang_text = inp_file.read()
            labse_sim = labse_detector.count_similiarity(src_lang_text, dst_lang_text)
            tfidf_sim = tfidf_detector.count_similiarity(src_lang_text, dst_lang_text)
            shingles_sim = shingles_detector.count_similiarity(src_lang_text, dst_lang_text)
            print(f'{filename}:{dst_lang_file_path},{acc},{labse_sim},{tfidf_sim}{shingles_sim}')
            # progress_bar.next()
    # progress_bar.finish()


def main():
    with open(common.SHUFFLED_TEXTS.joinpath('originality')) as inp_file:
        originality = eval(inp_file.read())
    print(originality)

if __name__ == '__main__':
    main()
