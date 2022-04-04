import argparse
import random

import consts
from scripts.detectors.labse import definition as labse_definition
from scripts.detectors.tfidf import definition as tfifd_definition
from scripts.detectors.shingles import definition as shingles_definition
from scripts.prepare_texts import prepare_wikipedia_texts
from scripts.prepare_texts import shuffle_texts as shuffle_prepared_texts

from progress import bar

MAX_DST_FILES = 10
DIFFERENT_FILES_NUM = 100

def run_experiments(args):
    labse_detector = labse_definition.LabseDetector()
    tfidf_detector = tfifd_definition.TfIdfDetector()
    shingles_detector = shingles_definition.ShinglesDetector()
    with open(consts.SHUFFLED_TEXTS.joinpath('originality')) as inp_file:
        originality = eval(inp_file.read())
    print('Preparing completed!')
    src_texts_paths = [path for path in consts.SHUFFLED_TEXTS.glob(f'{consts.SRC_LANG}/*')][:10]
    # progress_bar = bar.IncrementalBar('Comparison', max=len(src_texts_paths))
    different_files_cntr = 0
    for src_lang_file_path in src_texts_paths:
        src_file_name = src_lang_file_path.stem
        with open(src_lang_file_path) as inp_file:
            src_lang_text = inp_file.read()
        comparisons_num = 0
        dst_files = set()
        while comparisons_num < MAX_DST_FILES:
            dst_file_name = None
            while dst_file_name is None or dst_file_name in dst_files:
                dst_file_name = random.choice(list(originality.keys()))
            dst_files.add(dst_file_name)
            intersection = set(originality[src_file_name]).intersection(originality[dst_file_name])
            if not intersection:
                if different_files_cntr > DIFFERENT_FILES_NUM:
                    continue
                different_files_cntr += 1
            with open(consts.SHUFFLED_TEXTS.joinpath(f'{consts.DST_LANG}/{dst_file_name}')) as inp_file:
                dst_lang_text = inp_file.read()
            labse_sim = labse_detector.count_similiarity(src_lang_text, dst_lang_text)
            tfidf_sim = tfidf_detector.count_similiarity(src_lang_text, dst_lang_text)
            shingles_sim = shingles_detector.count_similiarity(src_lang_text, dst_lang_text)
            acc = sum([originality[src_file_name][part] for part in intersection])
            acc, labse_sim, tfidf_sim, shingles_sim = [
                str(x).replace('.', ',') for x in [acc, labse_sim, tfidf_sim, shingles_sim]
            ]
            print(f'{src_file_name}:{dst_file_name};{acc};{labse_sim};{tfidf_sim};{shingles_sim}')
            #print(f'{src_file_name}:{dst_file_name},{acc},{shingles_sim}')
            comparisons_num += 1
            # progress_bar.next()
    # progress_bar.finish()


def prepare_texts(args):
    prepare_wikipedia_texts.main()


def shuffle_texts(args):
    shuffle_prepared_texts.main()
    

def main():
    parser = argparse.ArgumentParser(description='Compare different methods of xlplagiarism')
    parser.add_argument('--prepare-texts',  action='store_true')
    parser.add_argument('--shuffle-texts', action='store_true')
    parser.add_argument('--run-experiments', action='store_true')
    args = parser.parse_args()

    if args.prepare_texts:
        prepare_texts(args)
    if args.shuffle_texts:
        shuffle_texts(args)
    if args.run_experiments:
        run_experiments(args)

if __name__ == '__main__':
    main()
