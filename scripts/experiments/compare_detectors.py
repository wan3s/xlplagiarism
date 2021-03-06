import random
import time

from scripts import consts
from scripts.detectors.labse import definition as labse_definition
from scripts.detectors.tfidf import definition as tfifd_definition
from scripts.detectors.shingles import definition as shingles_definition


MAX_DST_FILES = 2  # сколько выбираем файлов на dst языке для очередного файла на src языке
DIFFERENT_FILES_NUM = 2  # число непересекающихся текстов для всего эксперимента


def run_experiments(args):
    labse_detector = labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold)
    tfidf_detector = tfifd_definition.TfIdfDetector()
    shingles_detector = shingles_definition.ShinglesDetector()
    with open(consts.SHUFFLED_TEXTS.joinpath('originality')) as inp_file:
        originality = eval(inp_file.read())[args.dataset]
    print('Preparing completed!')
    shuffled_texts_dir = consts.SHUFFLED_TEXTS.joinpath(args.dataset)
    src_texts_paths = [path for path in shuffled_texts_dir.glob(f'{consts.SRC_LANG}/*')][:5]
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
            with open(consts.SHUFFLED_TEXTS.joinpath(f'{args.dataset}/{consts.DST_LANG}/{dst_file_name}')) as inp_file:
                dst_lang_text = inp_file.read()
            try:
                labse_sim = labse_detector.count_similiarity(src_lang_text, dst_lang_text)
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
