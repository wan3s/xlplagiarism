import collections
import deep_translator
import nltk
import typing as tp
import time

from progress import bar

from scripts import consts
from scripts.detectors import common
from scripts.detectors.labse import definition as labse_definition
from scripts.detectors.tfidf import definition as tfifd_definition
from scripts.detectors.shingles import definition as shingles_definition


def translate_text(translator, text):
    splitted = nltk.sent_tokenize(text)
    translated_text = [
        translate_sentense(translator, sentense)
        for sentense in splitted
    ]
    return ' '.join(translated_text)


def translate_sentense(translator, sentense):
    splitted = nltk.tokenize.word_tokenize(sentense)
    chunk_size = 100
    while chunk_size >= 1:
        translated_text = []
        try:

            if hasattr(translator, 'translate_batch'):
                translated_text = translator.translate_batch(
                    [text for text  in _by_chunks(splitted, int(chunk_size))]
                )
            elif hasattr(translator, 'translate_words'):
                translated_text = translator.translate_words(splitted)
            else:
                for word in splitted:
                    translated_text.append(
                        translator.translate(word)
                    )
                    time.sleep(1)
            break
        except deep_translator.exceptions.RequestError:
            chunk_size /= 5
    else:
        raise RuntimeError('Doesn\'t translated')
    return ' '.join(translated_text)


def _by_chunks(arr, chunk_size):
    res = []
    while arr:
        res.append(' '.join(arr[:chunk_size]))
        arr = arr[chunk_size:]
    return res


def check_translators_for_detector(
    args,
    detector: common.BaseDetector, 
    translator: deep_translator.base.BaseTranslator,
    experiment_name: tp.Optional[str] = None,
):
    shuffled_texts_dir = consts.SHUFFLED_TEXTS.joinpath(args.dataset)
    texts_paths = [path for path in shuffled_texts_dir.glob(f'{consts.SRC_LANG}/*')][:20]
    result = {}
    progress_bar = bar.IncrementalBar(experiment_name or 'Experiment', max=len(texts_paths))
    for text_path in texts_paths:
        progress_bar.next()
        filename = text_path.stem
        with open(text_path, 'r') as inp_file:
            src_lang_text = inp_file.read()
        with open(consts.SHUFFLED_TEXTS.joinpath(f'{args.dataset}/{consts.DST_LANG}/{filename}'), 'r') as inp_file:
            dst_lang_text = inp_file.read()
        translated_src_lang_text = translate_text(translator, dst_lang_text)
        result[filename] = str(detector.count_similiarity(src_lang_text, translated_src_lang_text)).replace('.', ',')
    progress_bar.finish()
    return result


def run(args):
    total_result = collections.defaultdict(list)
    head = ['filename']
    detectors = [
        (labse_definition.LabseDetector(sim_threshold=args.labse_sim_threshold), 'labse'),
        (tfifd_definition.TfIdfDetector(translate=False), 'tfidf'),
        (shingles_definition.ShinglesDetector(translate=False), 'shingles'),
    ]
    translators = [
        (
            deep_translator.GoogleTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
            'google',
        ),
        # (
        #     deep_translator.MyMemoryTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
        #     'my_memory',
        # ),
        # (
        #     deep_translator.PonsTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
        #     'pons',
        # ),
        # (
        #     deep_translator.LingueeTranslator(source=consts.DST_LANG, target=consts.SRC_LANG),
        #     'linguee',
        # ),
    ]
    for detector, detector_name in detectors:
        for translator, translator_name in translators:
            exp_name = f'{detector_name}-{translator_name}'
            head.append(exp_name)
            result = check_translators_for_detector(args, detector, translator, exp_name)
            for filename, coef in result.items():
                total_result[filename].append(coef)
    with open(f'{args.outfile_name}.tsv', 'a') as out_file:
        out_file.write('\t'.join(head))
        out_file.write('\n')
        for filename, row in total_result.items():
            coefs = '\t'.join(row)
            out_file.write(f'{filename}\t{coefs}\n')
            

        

