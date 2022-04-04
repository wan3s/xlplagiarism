import pathlib

from detectors.labse import definition as labse_definition
from detectors.tfidf import definition as tfifd_definition
from detectors.shingles import definition as shingles_definition
from detectors import common

from progress import bar

import deep_translator

MAX_DST_FILES = 10
DIFFERENT_FILES_NUM = 100

def main():
    labse_detector = labse_definition.LabseDetector()
    tfidf_detector = tfifd_definition.TfIdfDetector()
    shingles_detector = shingles_definition.ShinglesDetector()
    with open(common.SHUFFLED_TEXTS.joinpath('originality')) as inp_file:
        originality = eval(inp_file.read())
    print('Preparing completed!')
    src_texts_paths = [path for path in common.SHUFFLED_TEXTS.glob(f'{common.SRC_LANG}/*')][:10]
    # progress_bar = bar.IncrementalBar('Comparison', max=len(src_texts_paths))
    different_files_cntr = 0
    for src_lang_file_path in src_texts_paths:
        src_file_name = src_lang_file_path.stem
        with open(src_lang_file_path) as inp_file:
            src_lang_text = inp_file.read()
        comparisons_num = 0
        for dst_file_name in originality.keys():
            if comparisons_num >= MAX_DST_FILES:
                break
            intersection = set(originality[src_file_name]).intersection(originality[dst_file_name])
            if not intersection:
                if different_files_cntr > DIFFERENT_FILES_NUM:
                    continue
                different_files_cntr += 1
            with open(common.SHUFFLED_TEXTS.joinpath(f'{common.DST_LANG}/{dst_file_name}')) as inp_file:
                dst_lang_text = inp_file.read()
            labse_sim = labse_detector.count_similiarity(src_lang_text, dst_lang_text)
            tfidf_sim = tfidf_detector.count_similiarity(src_lang_text, dst_lang_text)
            shingles_sim = shingles_detector.count_similiarity(src_lang_text, dst_lang_text)
            acc = sum([originality[src_file_name][part] for part in intersection])
            print(f'{src_file_name}:{dst_file_name},{acc},{labse_sim},{tfidf_sim},{shingles_sim}')
            #print(f'{src_file_name}:{dst_file_name},{acc},{shingles_sim}')
            comparisons_num += 1
            # progress_bar.next()
    # progress_bar.finish()


if __name__ == '__main__':
    main()
