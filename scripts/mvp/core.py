import nltk
import pathlib

from scripts import consts
from scripts.detectors.labse import definition as labse_definition


def run_program(args):
    if not args.input_file:
        print('Please pass input file using --input-file key')
        return
    labse_detector = labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold)
    with open(args.input_file, 'r') as inp_file:
        raw_text = inp_file.read()

    dst_lang_texts = traverse_texts(consts.TRANSLATED_TEXTS.joinpath(consts.DST_LANG))
    total_sim_sentences = []
    raw_text_sentences = [x for x in nltk.sent_tokenize(raw_text) if x]
    total_coef = 1
    for dst_lang_text in dst_lang_texts:
        if not raw_text_sentences:
            break
        coef, sim_sentences_text = labse_detector.get_sim_sentences(
            '. '.join(raw_text_sentences), 
            dst_lang_text,
        )
        total_coef *= (1 - coef)
        sim_sentences = [x for x in nltk.sent_tokenize(sim_sentences_text) if x]
        raw_text_sentences = [x for x in raw_text_sentences if x not in sim_sentences]
        total_sim_sentences += sim_sentences
    
    if total_coef >= 0.7:
        print(f'Originality={total_coef}: these texts are completely different')
    elif total_coef >= 0.3:
        print(f'Originality={total_coef}: these texts may be have some common quotes')
    else:
        print(f'Originality={total_coef}: are these text really different?')
    if total_sim_sentences:
        print('These sentences are very similar:')
        print('\n'.join(total_sim_sentences))


def traverse_texts(root_dir: pathlib.Path) -> str:
    res = {}
    for txt_file in root_dir.glob('*'):
        with open(txt_file) as inp_file:
            res[txt_file.stem] = inp_file.read()
    return res