import nltk
import typing as tp
import numpy as np

from sentence_transformers import SentenceTransformer

from scripts.detectors import common

MODEL_URL = 'sentence-transformers/LaBSE'

class LabseDetector(common.BaseDetector):
  def __init__(self, *args, sim_threshold, **kwargs):
    print('LabseDetector initializing ...')
    super().__init__(*args, **kwargs)
    self._sim_threshold = sim_threshold
    print(f'LabseDetector sim threshold: {self._sim_threshold}')
    self._model = SentenceTransformer(MODEL_URL)

  def count_similiarity_whole_text(self, src_lang_text: str, dst_lang_text: str):
    src_lang_embeddings, dst_lang_embeddings = [
      self._model.encode(text) for text in [[src_lang_text], [dst_lang_text]]
    ]
    return np.diag(np.matmul(dst_lang_embeddings, np.transpose(src_lang_embeddings)))[0]

  def count_similiarity(self, src_lang_text: str, dst_lang_text: str):
    src_sentences = nltk.sent_tokenize(src_lang_text)
    dst_sentences = nltk.sent_tokenize(dst_lang_text)
    src_lang_embeddings = self._model.encode(src_sentences)
    dst_lang_embeddings = self._model.encode(dst_sentences)
    res = np.matmul(dst_lang_embeddings, np.transpose(src_lang_embeddings))
    sim_sentences = []
    for j in range(len(src_sentences)):
      max_sim = 0
      for i in range(len(dst_sentences)):
        max_sim = max(max_sim, res[i][j])
      if max_sim >= self._sim_threshold:
        sim_sentences.append(j)
    sim_sentences_text = ' '.join(
      src_sentences[idx] for idx in sim_sentences
    )
    return len(nltk.tokenize.word_tokenize(sim_sentences_text)) / len(nltk.tokenize.word_tokenize(src_lang_text))
