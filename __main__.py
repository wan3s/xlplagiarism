import argparse
import random
import time

from scripts import consts
from scripts.compare_translators import main as compare_translators
from scripts.detectors.labse import definition as labse_definition
from scripts.detectors.tfidf import definition as tfifd_definition
from scripts.detectors.shingles import definition as shingles_definition
from scripts.prepare_texts import prepare_wikipedia_texts
from scripts.prepare_texts import shuffle_texts as shuffle_prepared_texts

from progress import bar

MAX_DST_FILES = 2  # сколько выбираем файлов на dst языке для очередного файла на src языке
DIFFERENT_FILES_NUM = 2  # число непересекающихся текстов для всего эксперимента

def run_experiments(args):
    labse_detector = labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold)
    tfidf_detector = tfifd_definition.TfIdfDetector()
    shingles_detector = shingles_definition.ShinglesDetector()
    with open(consts.SHUFFLED_TEXTS.joinpath('originality')) as inp_file:
        originality = eval(inp_file.read())
    print('Preparing completed!')
    src_texts_paths = [path for path in consts.SHUFFLED_TEXTS.glob(f'{consts.SRC_LANG}/*')][:5]
    different_files_cntr = 0
    res_output = []
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
            try:
                labse_sim = labse_detector.count_similiarity_parts(src_lang_text, dst_lang_text)
                tfidf_sim = tfidf_detector.count_similiarity(src_lang_text, dst_lang_text)
                shingles_sim = shingles_detector.count_similiarity(src_lang_text, dst_lang_text)
            except BaseException:
                print('Exception gotten => continue')
                time.sleep(5)
                continue
            acc = sum([originality[src_file_name][part] for part in intersection])
            acc, labse_sim, tfidf_sim, shingles_sim = [
                str(x).replace('.', ',') for x in [acc, labse_sim, tfidf_sim, shingles_sim]
            ]
            res = f'{src_file_name}:{dst_file_name};{acc};{labse_sim};{tfidf_sim};{shingles_sim}'
            res_output.append(res)
            print(res)
            comparisons_num += 1
    with open(f'{args.outfile_name}.csv', 'w') as out_file:
        out_file.write('\n'.join(res_output))
    

def run_program(args):
    if not args.input_files:
        print('Please pass two input files using --input-files key')
        return
    labse_detector = labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold)
    inp_file_name1, inp_file_name2 = args.input_files
    with open(inp_file_name1, 'r') as inp_file:
        raw_text1 = inp_file.read()
    with open(inp_file_name2, 'r') as inp_file:
        raw_text2 = inp_file.read()
    coef, sim_sentences = labse_detector.get_sim_sentences(raw_text1, raw_text2)
    originality = 1-coef
    if coef < 0.3:
        print(f'Originality={originality}: these texts are completely different')
    elif coef < 0.7:
        print(f'Originality={originality}: these texts may be have some common quotes')
    else:
        print(f'Originality={originality}: are these text really different?')
    if sim_sentences:
        print('These sentences are very similar:')
        print(sim_sentences)


def main():
    parser = argparse.ArgumentParser(description='Compare different methods of xlplagiarism')
    parser.add_argument('--prepare-texts',  action='store_true')
    parser.add_argument('--shuffle-texts', action='store_true')
    parser.add_argument('--run-experiments', action='store_true')
    parser.add_argument('--compare-translators', action='store_true')
    parser.add_argument('--run-program', action='store_true')
    parser.add_argument('--outfile-name', default='result')
    parser.add_argument('--input-files', nargs=2)
    parser.add_argument('--labse-sim-threshold', default=consts.LABSE_SIM_THRESHOLD, type=float)
    args = parser.parse_args()

    if args.prepare_texts:
        print('Preparing texts started ...')
        prepare_wikipedia_texts.main()
    if args.shuffle_texts:
        print('Shuffling texts strated ...')
        shuffle_prepared_texts.main()
    if args.run_experiments:
        print('Running experiments strated ...')
        run_experiments(args)
    if args.compare_translators:
        print('Comparing translators strated ...')
        compare_translators.run(args)
    if args.run_program:
        print('Running program ...')
        run_program(args)

if __name__ == '__main__':
    main()
