import bert
import typing as tp

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer

from detectors import common

MODEL_URL = 'https://tfhub.dev/google/LaBSE/1'

class LabseDetector(common.BaseDetector):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_seq_length = 64
    self.labse_model, self.labse_layer = self._get_model(
      model_url=MODEL_URL 
    )
    vocab_file = self.labse_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = self.labse_layer.resolved_object.do_lower_case.numpy()
    self.tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

  def count_similiarity(self, src_lang_text: str, dst_lang_text: str):
    src_lang_embeddings, dst_lang_embeddings = [
      self._encode(text) for text in [[src_lang_text], [dst_lang_text]]
    ]
    return np.diag(np.matmul(dst_lang_embeddings, np.transpose(src_lang_embeddings)))

  def encode(self, input_text):
    input_ids, input_mask, segment_ids = self._create_input(input_text)
    return self.labse_model([input_ids, input_mask, segment_ids])

  def _create_input(self, input_strings):
    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in input_strings:
      # Tokenize input.
      input_tokens = ["[CLS]"] + self.tokenizer.tokenize(input_string) + ["[SEP]"]
      input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
      sequence_length = min(len(input_ids), self.max_seq_length)

      # Padding or truncation.
      if len(input_ids) >= self.max_seq_length:
        input_ids = input_ids[:self.max_seq_length]
      else:
        input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))

      input_mask = [1] * sequence_length + [0] * (self.max_seq_length - sequence_length)

      input_ids_all.append(input_ids)
      input_mask_all.append(input_mask)
      segment_ids_all.append([0] * self.max_seq_length)
    return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)

  def _get_model(self, model_url):
    labse_layer = hub.KerasLayer(model_url, trainable=True)

    # Define input.
    input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                          name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                      name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    # LaBSE layer.
    pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

    # The embedding is l2 normalized.
    pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

    # Define model.
    return tf.keras.Model(
          inputs=[input_word_ids, input_mask, segment_ids],
          outputs=pooled_output), labse_layer
