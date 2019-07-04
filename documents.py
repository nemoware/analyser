import os

import numpy as np

from text_tools import Tokens, my_punctuation

TEXT_PADDING_SYMBOL = ' '

import sentencepiece as spm
import sentencepiece_pb2


class GTokenizer:
  def tokenize(self, s) -> Tokens:
    raise NotImplementedError()

  def untokenize(self, t: Tokens) -> str:
    raise NotImplementedError()


class SpmGTokenizer(GTokenizer):

  def __init__(self):
    super().__init__()
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    self.sp = spm.SentencePieceProcessor()
    p = os.path.join(__location__, 'vocab', 'm.model')
    print('loading tokenization model', p)
    self.sp.load(p)
    # self.spt = sentencepiece_pb2.SentencePieceText()

    # self.sp.set_encode_extra_options('bos:eos')
    # self.sp.set_decode_extra_options('bos:eos')

    # # encode: text => id
    # print(sp.encode_as_pieces('Лихо Рыбу мыл Вадим'))
    # print(sp.encode_as_ids('Лихо Рыбу мыл Вадим'))

  def tokenize(self, text) -> Tokens:
    # # return self.sp.encode_as_ids(text)
    # sentences = text.split('\n')
    # result = []
    # for i in range(len(sentences)):
    #   sentence = sentences[i]
    #   result += self.sp.encode_as_pieces(sentence)
    #   if i < len(sentences) - 1:
    #     result += ['\n']
    #
    # return result

    _txt_bytes = text.encode('utf-8')
    spt = sentencepiece_pb2.SentencePieceText()
    spt.ParseFromString(self.sp.encode_as_serialized_proto(text))  # Full width hello

    tokens = []

    for p in spt.pieces:
      b = p.begin
      e = p.end
      token = _txt_bytes[b:e].decode()
      tokens.append(token)

    return tokens

  def untokenize(self, t: Tokens) -> str:
    return ''.join(t)  # self.sp.decode_pieces(t)
    # pieces(t)


import nltk


class DefaultGTokenizer(GTokenizer):

  def __init__(self):

    nltk.download('punkt')
    from nltk.tokenize import _treebank_word_tokenizer
    nltk_treebank_word_tokenizer = _treebank_word_tokenizer

  def tokenize(self, text) -> Tokens:
    sentences = text.split('\n')
    result = []
    for i in range(len(sentences)):
      sentence = sentences[i]
      result += nltk.word_tokenize(sentence)
      if i < len(sentences) - 1:
        result += ['\n']

    return result

  def untokenize(self, tokens: Tokens) -> str:
    return "".join([" " + i if not i.startswith("'") and i not in my_punctuation else i for i in tokens]).strip()


TOKENIZER_DEFAULT = SpmGTokenizer()


class TokenizedText:

  def __init__(self, tokenizer: GTokenizer = TOKENIZER_DEFAULT):
    super().__init__()

    self.filename = None
    self.tokens_cc = None
    self.tokens: Tokens = None
    self.tokenizer = tokenizer

  def get_len(self):
    return len(self.tokens)

  def untokenize(self):
    return self.tokenizer.untokenize(self.tokens)

  def untokenize_cc(self):
    return self.tokenizer.untokenize(self.tokens_cc)

  def tokenize(self, _txt):
    return self.tokenizer.tokenize(_txt)

  def trim(self, sl: slice):
    self.tokens = self.tokens[sl]
    if self.tokens_cc:
      self.tokens_cc = self.tokens_cc[sl]
    self.categories_vector = self.categories_vector[sl]


class EmbeddableText(TokenizedText):
  def __init__(self, tokenizer: GTokenizer = TOKENIZER_DEFAULT):
    super().__init__(tokenizer)
    self.embeddings = None


# build tokens map to char pos
def tokens_map(_tokens):
  _text = ''
  _map = []
  for t in _tokens:
    _map.append(len(_text))
    _text += t
  _map.append(len(_text))
  return _map, _text


def token_index_by_char_index(char_index, _map):
  for i in range(len(_map) - 1):
    a = _map[i]
    b = _map[i + 1]

    if a <= char_index < b:
      return i

  return 0


class MarkedDoc(TokenizedText):

  def __init__(self, tokens, categories_vector=None, tokenizer: GTokenizer = TOKENIZER_DEFAULT):
    super().__init__(tokenizer)

    self.tokens = np.array(tokens)
    if categories_vector is None:
      categories_vector = np.zeros(len(tokens))

    assert len(tokens) == len(categories_vector)
    self.categories_vector = np.array(categories_vector)

    self.tokens_map, self.text = tokens_map(self.tokens)

  def filter(self, filter_op):
    new_tokens = []
    new_categories_vector = []

    for i in range(self.get_len()):
      _tuple = filter_op(self.tokens[i], self.categories_vector[i])

      if _tuple is not None:
        new_tokens.append(_tuple[0])
        new_categories_vector.append(_tuple[1])

    self.tokens = new_tokens
    self.categories_vector = new_categories_vector

  def slice(self, sl: slice):
    new_tokens = self.tokens[sl]
    new_categories_vector = self.categories_vector[sl]
    return MarkedDoc(new_tokens, new_categories_vector)

  def token_index_by_char_index(self, char_index):
    return token_index_by_char_index(char_index, self.tokens_map)

  def set_region_value_by_char_indices(self, char_slice: slice, value):
    a = token_index_by_char_index(char_slice.start, self.tokens_map)
    b = token_index_by_char_index(char_slice.stop, self.tokens_map)

    ts = slice(a, b)
    self.categories_vector[ts] = value
