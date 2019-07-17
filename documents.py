from text_tools import untokenize

TEXT_PADDING_SYMBOL = ' '


class TokenizedText:
  def __init__(self):
    super().__init__()

    self.filename = None
    self.tokens_cc = None
    self.tokens: Tokens = None

  def get_len(self):
    return len(self.tokens)

  def untokenize(self):
    return untokenize(self.tokens)

  def untokenize_cc(self):
    return untokenize(self.tokens_cc)

  def concat(self, doc: "TokenizedText"):
    self.tokens += doc.tokens
    self.categories_vector += doc.categories_vector
    if self.tokens_cc:
      self.tokens_cc += doc.tokens_cc

  def trim(self, sl: slice):
    self.tokens = self.tokens[sl]
    if self.tokens_cc:
      self.tokens_cc = self.tokens_cc[sl]
    self.categories_vector = self.categories_vector[sl]


class EmbeddableText(TokenizedText):
  def __init__(self):
    super().__init__()
    self.embeddings = None


class MarkedDoc(TokenizedText):

  def __init__(self, tokens, categories_vector):
    super().__init__()
    assert len(tokens) == len(categories_vector)
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

TEXT_PADDING_SYMBOL = ' '


class GTokenizer:
  def tokenize(self, s) -> Tokens:
    raise NotImplementedError()

  def untokenize(self, t: Tokens) -> str:
    raise NotImplementedError()


import nltk


class DefaultGTokenizer(GTokenizer):

  def __init__(self):
    nltk.download('punkt')
    from nltk.tokenize import _treebank_word_tokenizer
    self.nltk_treebank_word_tokenizer = _treebank_word_tokenizer

  def tokenize_line(self, line):
    return [line[t[0]:t[1]] for t in self.nltk_treebank_word_tokenizer.span_tokenize(line)]

  def tokenize(self, text) -> Tokens:
    return [text[t[0]:t[1]] for t in self.tokens_map(text)]

  def untokenize(self, tokens: Tokens) -> str:
    #TODO: remove it!!
    return "".join([" " + i if not i.startswith("'") and i not in my_punctuation else i for i in tokens]).strip()

  # build tokens map to char pos
  def tokens_map(self, text):

    result = []
    for i in range(len(text)):
      if text[i] == '\n':
        result.append([i, i + 1])

    result += [s for s in self.nltk_treebank_word_tokenizer.span_tokenize(text)]

    result.sort(key=lambda x: x[0])
    return result

#TODO: use it!
TOKENIZER_DEFAULT = DefaultGTokenizer()

if __name__ == '__main__':
  text = 'мама молилась Раме\n\nРама -- Вишну, в Вишну ел... черешню?'
  tts = TOKENIZER_DEFAULT.tokens_map(text)
  for t in tts:
    print(t)

  print(TOKENIZER_DEFAULT.tokenize(text))
