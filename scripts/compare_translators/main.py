import collections
import deep_translator
import typing as tp

from progress import bar

from scripts import consts
from scripts.detectors import common
from scripts.detectors.labse import definition as labse_definition
from scripts.detectors.tfidf import definition as tfifd_definition
from scripts.detectors.shingles import definition as shingles_definition


def check_translators_for_detector(
    detector: common.BaseDetector, 
    translator: deep_translator.BaseTranslator,
    experiment_name: tp.Optional[str] = None,
):
    texts_paths = [path for path in consts.SHUFFLED_TEXTS.glob(f'{consts.SRC_LANG}/*')]
    result = {}
    progress_bar = bar.IncrementalBar(experiment_name or 'Experiment', max=len(texts_paths))
    for text_path in texts_paths:
        filename = text_path.stem
        with open(text_path, 'r') as inp_file:
            src_lang_text = inp_file.read()
        with open(consts.SHUFFLED_TEXTS.joinpath(f'{consts.DST_LANG}/{filename}'), 'r') as inp_file:
            dst_lang_text = inp_file.read()
        translated_src_lang_text = common.translate_text(translator, dst_lang_text)
        result[filename] = detector.count_similiarity(src_lang_text, translated_src_lang_text)
        progress_bar.next()
    progress_bar.finish()


def run(args):
    total_result = collections.defaultdict(list)
    head = []
    detectors = [
        (labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold), 'labse'),
        (tfifd_definition.TfIdfDetector(), 'tfidf'),
        (shingles_definition.ShinglesDetector(), 'shingles'),
    ]
    translators = [
        (
            deep_translator.GoogleTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
            'google',
        ),
        (
            deep_translator.MyMemoryTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
            'my_memory',
        ),
        (
            deep_translator.LingueeTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
            'linguee',
        ),
        (
            deep_translator.PonsTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
            'pons',
        ),
    ]
    for detector, detector_name in detectors:
        for translator, translator_name in translators:
            exp_name = f'{detector_name}-{translator_name}'
            head.append(exp_name)
            result = check_translators_for_detector(detector, translator, exp_name)
            for filename, coef in result.items():
                total_result[filename].append(coef)
    print(head)
    for filename, row in total_result.items():
        coefs = '\t'.join(row)
        print(f'{filename}\t{coefs}')
            

        

