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
        ('sentences', consts.SENTENCE_MASKS, consts.CONF_PAPERS_SENTENCES_EN, consts.CONF_PAPERS_SENTENCES_FR),
        ('chunks', consts.CHUNK_MASKS, consts.CONF_PAPERS_CHUNKS_EN, consts.CONF_PAPERS_CHUNKS_FR),
    ]
    for level, masks_dir, en_dir, fr_dir in masks_and_data:
        total_tn_fp_fn_tp = [0, 0, 0, 0]
        for masks in [x for x in masks_dir.glob('*')][:1]:
            tn_fp_fn_tp = [0, 0, 0, 0]
            with open(masks, 'r') as mask_file:
                mask_file_lines = [x for x in mask_file.read().split('\n') if x][:10000]
            contents_by_filenames = {}
            loaded_masks = []
            progress_bar_content_coll = bar.IncrementalBar(f'Collecting content on {level}-level', max=len(mask_file_lines))
            for mask_file_line in mask_file_lines:
                mask = json.loads(mask_file_line)
                fr_file_template = make_file_name(mask['2'])
                en_file_template = make_file_name(mask['3'])
                fr_text_file = fr_file_template + '-fr.txt'
                en_text_file = en_file_template + '-en.txt'
                try:
                    with open(fr_dir.joinpath(fr_text_file), 'r') as fr_inp_file:
                        contents_by_filenames[fr_text_file] = fr_inp_file.read()
                    with open(en_dir.joinpath(en_text_file), 'r') as en_inp_file:
                        contents_by_filenames[en_text_file] = en_inp_file.read()
                except FileNotFoundError:
                    # print('File not found => continue')
                    pass
                
                loaded_masks.append(
                    {
                        'equal': int(fr_file_template == en_file_template),
                        'fr': fr_text_file,
                        'en': en_text_file,
                    }
                )
                progress_bar_content_coll.next()
            progress_bar_content_coll.finish()
                
            without_threads_start = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                progress_bar_comp = bar.IncrementalBar(f'Comparing on {level}-level', max=len(loaded_masks))
                for mask in loaded_masks:
                    fr_text_file, en_text_file = mask['fr'], mask['en']
                    progress_bar_comp.next()
                    def _worker(fr_text_file, en_text_file):
                        fr_text, en_text = contents_by_filenames[fr_text_file], contents_by_filenames[en_text_file]
                        y = int(labse_detector.count_similiarity_whole_text(fr_text, en_text) > labse_detector.sim_threshold)
                        idx = y + mask['equal'] * 2
                        return idx
                    futures.append(executor.submit(_worker, fr_text_file, en_text_file))
                progress_bar_comp.finish()
                progress_bar_res = bar.IncrementalBar(f'Result on {level}-level', max=len(loaded_masks))
                for future in concurrent.futures.as_completed(futures):
                    progress_bar_res.next()
                    idx = future.result()
                    tn_fp_fn_tp[idx] += 1
                    total_tn_fp_fn_tp[idx] += 1
                progress_bar_res.finish()
            print(f'\ntn_fp_fn_tp: {tn_fp_fn_tp}')
            print(time.time() - without_threads_start)
        print(f'\nTOTAL tn_fp_fn_tp: {total_tn_fp_fn_tp}')


def make_file_name(file_name):
    file_name_arr = file_name.split('-')
    if len(file_name_arr) > 3:
        file_name_arr = file_name_arr[:-1]
    return '-'.join(file_name_arr)
