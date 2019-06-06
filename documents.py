from text_tools import Tokens, untokenize

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
