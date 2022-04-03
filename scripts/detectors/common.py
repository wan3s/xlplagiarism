import nltk

class BaseDetector:
    def count_similiarity(self, src_lang_text, dst_lang_text):
        raise NotImplementedError


def translate_text(translator, text):
    translated_text = []
    splitted = nltk.sent_tokenize(text)
    while splitted:
        translated_text.append(
            translator.translate(
                ' '.join(splitted[:5])
            )
        )
        splitted = splitted[5:]
    return ' '.join(translated_text)
