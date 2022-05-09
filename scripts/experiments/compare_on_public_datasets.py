import json

import time
import collections
import concurrent.futures

from progress import bar

from scripts import consts
from scripts.detectors.labse import definition as labse_definition


def run(args):
    labse_detector = labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold)
    masks_and_data = [
        ('documents', consts.DOCS_MASKS, consts.CONF_PAPERS_DOCS_EN, consts.CONF_PAPERS_DOCS_FR),
        ('chunks', consts.CHUNK_MASKS, consts.CONF_PAPERS_CHUNKS_EN, consts.CONF_PAPERS_CHUNKS_FR),
        ('sentences', consts.SENTENCE_MASKS, consts.CONF_PAPERS_SENTENCES_EN, consts.CONF_PAPERS_SENTENCES_FR),
    ]
    for level, masks_dir, en_dir, fr_dir in masks_and_data:
        total_tn_fp_fn_tp = [0, 0, 0, 0]
        for masks in [x for x in masks_dir.glob('*')][:1]:
            tn_fp_fn_tp = [0, 0, 0, 0]
            with open(masks, 'r') as mask_file:
                mask_file_lines = [x for x in mask_file.read().split('\n') if x][:1000]
            dst_files_by_src_files = collections.defaultdict(list)
            for mask_file_line in mask_file_lines:
                mask = json.loads(mask_file_line)
                fr_text_file = make_file_name(mask['2'], '-fr.txt')
                en_text_file = make_file_name(mask['3'], '-en.txt')
                dst_files_by_src_files[fr_text_file].append(en_text_file)
            without_threads_start = time.time()
            progress_bar = bar.IncrementalBar(f'Comparing on {level}-level', max=len(dst_files_by_src_files))
            for fr_text_file, en_text_files in dst_files_by_src_files.items():
                with open(fr_dir.joinpath(fr_text_file), 'r') as fr_inp_file:
                    fr_text = fr_inp_file.read()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for en_text_file in en_text_files:
                        def _worker(en_text_file):
                            with open(en_dir.joinpath(en_text_file), 'r') as en_inp_file:
                                en_text = en_inp_file.read()
                            etal = int(mask['0'] == mask['1'])
                            y = int(labse_detector.count_similiarity_whole_text(fr_text, en_text) > labse_detector.sim_threshold)
                            idx = y + etal * 2
                            return idx
                        futures.append(executor.submit(_worker, en_text_file))
                    for future in concurrent.futures.as_completed(futures):
                        idx = future.result()
                        tn_fp_fn_tp[idx] += 1
                        total_tn_fp_fn_tp[idx] += 1
                progress_bar.next()
            print(time.time() - without_threads_start)
            print(f'\ntn_fp_fn_tp: {tn_fp_fn_tp}')
            progress_bar.finish()
        print(f'\nTOTAL tn_fp_fn_tp: {total_tn_fp_fn_tp}')


def make_file_name(file_name, postfix):
    file_name_arr = file_name.split('-')
    if len(file_name_arr) > 3:
        file_name_arr = file_name_arr[:-1]
    return '-'.join(file_name_arr) + postfix
