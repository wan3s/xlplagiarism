import typing as tp

import numpy as np

from sentence_transformers import SentenceTransformer

from detectors import common

MODEL_URL = 'sentence-transformers/LaBSE'

class LabseDetector(common.BaseDetector):
  def __init__(self, *args, **kwargs):
    print('LabseDetector initializing ...')
    super().__init__(*args, **kwargs)
    self._model = SentenceTransformer(MODEL_URL)

  def count_similiarity(self, src_lang_text: str, dst_lang_text: str):
    src_lang_embeddings, dst_lang_embeddings = [
      self._model.encode(text) for text in [[src_lang_text], [dst_lang_text]]
    ]
    return np.diag(np.matmul(dst_lang_embeddings, np.transpose(src_lang_embeddings)))[0]
