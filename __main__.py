import argparse

from scripts import consts
from scripts.experiments import compare_detectors
from scripts.experiments import compare_on_public_datasets
from scripts.experiments import compare_translators
from scripts.prepare_texts import prepare_wikipedia_texts
from scripts.prepare_texts import shuffle_texts as shuffle_prepared_texts


def main():
    parser = argparse.ArgumentParser(description='Compare different methods of xlplagiarism')
    parser.add_argument('--prepare-texts',  action='store_true')
    parser.add_argument('--shuffle-texts', action='store_true')
    parser.add_argument('--run-experiments', action='store_true')
    parser.add_argument('--compare-translators', action='store_true')
    parser.add_argument('--run-on-public-datasets', action='store_true')
    parser.add_argument(
        '--dataset',
        default=consts.TEST_DATASET, 
        choices=consts.SHUFFLED_TEXTS_SUBDIRS,
    )
    parser.add_argument('--run-program', action='store_true')
    parser.add_argument('--outfile-name', default='result')
    parser.add_argument('--input-file', nargs=1)
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
    if args.run_on_public_datasets:
        print('Comparing methods on public datasets ...')
        compare_on_public_datasets.run(args)

if __name__ == '__main__':
    main()
