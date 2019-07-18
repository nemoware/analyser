from text_tools import untokenize

TEXT_PADDING_SYMBOL = ' '


class TextMap:

  def __init__(self, text: str, map=None):
    self.text = text
    if map is None:
      self.map = TOKENIZER_DEFAULT.tokens_map(self.text)
    else:
      self.map = map

    self.untokenize = self.text_range  # alias

  def text_range(self, span):
    start = self.map[span[0]][0]
    stop = self.map[span[1] - 1][1]

    # assume map is ordered
    return self.text[start: stop]

  def tokens_in_range(self, span):
    tokens_i = self.map[span[0]:span[1]]
    return [
      self.text[tr[0]:tr[1]] for tr in tokens_i
    ]

  def split_text(self, txt: str):
    return [
      txt[tr[0]:tr[1]] for tr in self.map
    ]

  def get_len(self):
    return len(self.map)

  def __len__(self):
    return self.get_len()

   

  def __getitem__(self, key):
    if isinstance(key, slice):
      # Get the start, stop, and step from the slice
      return [self[ii] for ii in range(*key.indices(len(self)))]
    elif isinstance(key, int):

      r = self.map[key]
      # print('__getitem__', key)
      return self.text[r[0]:r[1]]
    else:
      raise TypeError("Invalid argument type.")

  def get_tokens(self):
    return self.split_text(self.text)

  tokens = property(get_tokens)


import warnings


class TokenizedText:

  def __init__(self):
    warnings.warn("deprecated", DeprecationWarning)
    super().__init__()

    self.tokens_cc = None
    self.tokens: Tokens = None

  def get_len(self):
    return len(self.tokens)

  def untokenize(self):
    warnings.warn("deprecated", DeprecationWarning)
    return untokenize(self.tokens)

  def untokenize_cc(self):
    warnings.warn("deprecated", DeprecationWarning)
    return untokenize(self.tokens_cc)

  def concat(self, doc: "TokenizedText"):
    warnings.warn("deprecated", DeprecationWarning)
    self.tokens += doc.tokens
    self.categories_vector += doc.categories_vector
    if self.tokens_cc:
      self.tokens_cc += doc.tokens_cc

  def trim(self, sl: slice):
    warnings.warn("deprecated", DeprecationWarning)
    self.tokens = self.tokens[sl]
    if self.tokens_cc:
      self.tokens_cc = self.tokens_cc[sl]
    self.categories_vector = self.categories_vector[sl]


class EmbeddableText(TokenizedText):
  warnings.warn("deprecated", DeprecationWarning)

  def __init__(self):
    warnings.warn("deprecated", DeprecationWarning)
    super().__init__()
    self.embeddings = None


class MarkedDoc(TokenizedText):

  def __init__(self, tokens, categories_vector):
    super().__init__()

    self.tokens = tokens
    self.categories_vector = categories_vector

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


# ---------------------------------------------------

from text_tools import Tokens, my_punctuation


class GTokenizer:
  def tokenize(self, s) -> Tokens:
    raise NotImplementedError()

  def untokenize(self, t: Tokens) -> str:
    raise NotImplementedError()


import nltk


def span_tokenize(text):
  ix = 0
  for word_token in nltk.word_tokenize(text):
    ix = text.find(word_token, ix)
    end = ix + len(word_token)
    yield (ix, end)
    ix = end


class DefaultGTokenizer(GTokenizer):

  def __init__(self):
    nltk.download('punkt')

  def tokenize_line(self, line):
    return [line[t[0]:t[1]] for t in span_tokenize(line)]

  def tokenize(self, text) -> Tokens:
    return [text[t[0]:t[1]] for t in self.tokens_map(text)]

  def untokenize(self, tokens: Tokens) -> str:
    # TODO: remove it!!
    return "".join([" " + i if not i.startswith("'") and i not in my_punctuation else i for i in tokens]).strip()

  # build tokens map to char pos
  def tokens_map(self, text):

    result = []
    for i in range(len(text)):
      if text[i] == '\n':
        result.append([i, i + 1])

    result += [s for s in span_tokenize(text)]

    result.sort(key=lambda x: x[0])
    return result


# TODO: use it!
TOKENIZER_DEFAULT = DefaultGTokenizer()
