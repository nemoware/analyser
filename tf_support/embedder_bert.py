import bert

import tensorflow_hub as hub


from analyser.embedding_tools import AbstractEmbedder

from tensorflow import keras
from tensorflow.keras import layers

b_instance = None


class BertEmbedder(AbstractEmbedder):

  @staticmethod
  def get_instance():
    global b_instance
    if b_instance is None:
      e = BertEmbedder()
      b_instance = e
    return b_instance

  def __init__(self):
    self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2", trainable=False)
    self.vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
    self.do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()

    self.tokenizer = bert.bert_tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)

  def embedd_tokenized_sentences(self, tokenized_sentences):
    max_seq_length: int = len(max(tokenized_sentences, key=len))

    print('max_seq_length=', max_seq_length, 'sentences=', len(tokenized_sentences))
    input_ids_, input_masks_, input_segments_ = self.encode_tokens_for_embedding(tokenized_sentences, max_seq_length)

    input_word_ids = layers.Input(shape=(max_seq_length,), dtype='int32', name="input_word_ids")
    input_mask = layers.Input(shape=(max_seq_length,), dtype='int32', name="input_mask")
    segment_ids = layers.Input(shape=(max_seq_length,), dtype='int32', name="segment_ids")

    pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])

    model = keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

    pool_embs, all_embs = model.predict([input_ids_, input_masks_, input_segments_])

    return pool_embs, all_embs

  def get_masks(self, tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
      raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))

  def get_segments(self, tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
      raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
      segments.append(current_segment_id)
      if token == "[SEP]":
        current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

  def get_ids(self, tokens, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids

  def tokenize_sentences(self, sentences):
    stokens_ = []

    for line in sentences:
      ltokens = self.tokenizer.tokenize(line)
      ltokens = ["[CLS]"] + ltokens + ["[SEP]"]
      stokens_.append(ltokens)

    return stokens_

  def encode_tokens_for_embedding(self, tokenized_sentences, max_seq_length):

    input_ids_ = [self.get_ids(tt, max_seq_length) for tt in tokenized_sentences]
    input_masks_ = [self.get_masks(tt, max_seq_length) for tt in tokenized_sentences]
    input_segments_ = [self.get_segments(tt, max_seq_length) for tt in tokenized_sentences]

    return input_ids_, input_masks_, input_segments_
