import nltk
import deep_translator

_CHUNK_SIZE = 5

class BaseDetector:
    def count_similiarity(self, src_lang_text, dst_lang_text):
        raise NotImplementedError


def translate_text(translator, text):
    splitted = nltk.sent_tokenize(text)
    chunk_size = _CHUNK_SIZE
    while chunk_size >= 1:
        translated_text = []
        try:
            for text in _by_chunks(splitted, int(chunk_size)):
                translated_text.append(
                    translator.translate(
                        ' '.join(text)
                    )
                )
            break
        except deep_translator.exceptions.RequestError:
            chunk_size /= 5
    else:
        raise RuntimeError('Doesn\'t translated')
    return ' '.join(translated_text)


def _by_chunks(arr, chunk_size):
    res = []
    while arr:
        res.append(arr[:chunk_size])
        arr = arr[chunk_size:]
    return res
