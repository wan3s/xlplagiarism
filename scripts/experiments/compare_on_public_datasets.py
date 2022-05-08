import json

from progress import bar

from scripts import consts
from scripts.detectors.labse import definition as labse_definition


def run(args):
    labse_detector = labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold)
    masks_and_data = [
        ('chunks', consts.CHUNK_MASKS, consts.CONF_PAPERS_CHUNKS_EN, consts.CONF_PAPERS_CHUNKS_FR),
        ('sentences', consts.SENTENCE_MASKS, consts.CONF_PAPERS_SENTENCES_EN, consts.CONF_PAPERS_SENTENCES_FR),
        ('documents', consts.DOCS_MASKS, consts.CONF_PAPERS_DOCS_EN, consts.CONF_PAPERS_DOCS_FR),
    ]
    for level, masks_dir, en_dir, fr_dir in masks_and_data:
        for masks in masks_dir.glob('*'):
            tn_fp_fn_tp = [0, 0, 0, 0]
            with open(masks, 'r') as mask_file:
                mask_file_lines = [x for x in mask_file.read().split('\n') if x]
            progress_bar = bar.IncrementalBar(f'Comparing on {level}-level', max=len(mask_file_lines))
            for mask_file_line in mask_file_lines:
                mask = json.loads(mask_file_line)
                fr_text_file = make_file_name(mask['2'], '-fr.txt')
                en_text_file = make_file_name(mask['3'], '-en.txt')
                try:
                    with open(fr_dir.joinpath(fr_text_file), 'r') as fr_inp_file:
                        fr_text = fr_inp_file.read()
                    with open(en_dir.joinpath(en_text_file), 'r') as en_inp_file:
                        en_text = en_inp_file.read()
                except FileNotFoundError:
                    print('File not found => continue')
                etal = int(mask['0'] == mask['1'])
                y = int(labse_detector.count_similiarity_whole_text(fr_text, en_text) > labse_detector.sim_threshold)
                tn_fp_fn_tp[y + etal * 2] += 1
                progress_bar.next()
            print(f'\ntn_fp_fn_tp: {tn_fp_fn_tp}')
            progress_bar.finish()


def make_file_name(file_name, postfix):
    file_name_arr = file_name.split('-')
    if len(file_name_arr) > 3:
        file_name_arr = file_name_arr[:-1]
    return '-'.join(file_name_arr) + postfix
