import argparse

from scripts import consts
from scripts.experiments import compare_detectors
from scripts.experiments import compare_translators
from scripts.detectors.labse import definition as labse_definition
from scripts.prepare_texts import prepare_wikipedia_texts
from scripts.prepare_texts import shuffle_texts as shuffle_prepared_texts
    

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
    parser.add_argument(
        '--dataset',
        default=consts.TEST_DATASET, 
        choices=consts.SHUFFLED_TEXTS_SUBDIRS,
    )
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
        compare_detectors.run_experiments(args)
    if args.compare_translators:
        print('Comparing translators strated ...')
        compare_translators.run(args)
    if args.run_program:
        print('Running program ...')
        run_program(args)

if __name__ == '__main__':
    main()
