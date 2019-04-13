# legal_docs.py

import time
from functools import wraps
from typing import List

from doc_structure import DocumentStructure, headline_probability, StructureLine
from embedding_tools import embedd_tokenized_sentences_list
from ml_tools import normalize, smooth, relu, extremums, smooth_safe, remove_similar_indexes
from patterns import *
from text_normalize import *
from text_tools import *
from transaction_values import extract_sum_from_tokens

PROF_DATA = {}

REPORTED_DEPRECATED = {}


def deprecated(fn):
  @wraps(fn)
  @wraps(fn)
  def with_reporting(*args, **kwargs):
    if fn.__name__ not in REPORTED_DEPRECATED:
      REPORTED_DEPRECATED[fn.__name__] = 1
      print("----WARNING!: function {} is deprecated".format(fn.__name__))

    ret = fn(*args, **kwargs)
    return ret

  return with_reporting


def profile(fn):
  @wraps(fn)
  @wraps(fn)
  def with_profiling(*args, **kwargs):
    start_time = time.time()

    ret = fn(*args, **kwargs)

    elapsed_time = time.time() - start_time

    if fn.__name__ not in PROF_DATA:
      PROF_DATA[fn.__name__] = [0, []]
    PROF_DATA[fn.__name__][0] += 1
    PROF_DATA[fn.__name__][1].append(elapsed_time)

    return ret

  return with_profiling


def print_prof_data():
  for fname, data in PROF_DATA.items():
    max_time = max(data[1])
    avg_time = sum(data[1]) / len(data[1])
    print("Function {} called {} times. ".format(fname, data[0]))
    print('Execution time max: {:.4f}, average: {:.4f}'.format(max_time, avg_time))


def clear_prof_data():
  global PROF_DATA
  PROF_DATA = {}


class LegalDocument(EmbeddableText):

  def __init__(self, original_text=None):
    self.original_text = original_text
    self.filename = None
    self.tokens = None
    self.tokens_cc = None
    self.embeddings = None
    self.normal_text = None
    self.distances_per_pattern_dict = None

    self.right_padding = 0

    # subdocs
    self.start = None
    self.end = None

  def find_sum_in_section(self):
    raise Exception('not implemented')

  def find_sentence_beginnings(self, best_indexes):
    return [find_token_before_index(self.tokens, i, '\n', 0) for i in best_indexes]

  @profile
  def calculate_distances_per_pattern(self, pattern_factory: AbstractPatternFactory, dist_function=DIST_FUNC):
    distances_per_pattern_dict = {}
    for pat in pattern_factory.patterns:
      try:
        dists = pat._eval_distances_multi_window(self.embeddings, dist_function)
        if self.right_padding > 0:
          dists = dists[:-self.right_padding]
        # TODO: this inversion must be a part of a dist_function
        dists = 1.0 - dists
        distances_per_pattern_dict[pat.name] = dists
        dists.flags.writeable = False
      except Exception as e:
        print('ERROR: calculate_distances_per_pattern ', e)
        dists = np.zeros(len(self.embeddings))
        if self.right_padding > 0:
          dists = dists[:-self.right_padding]
        distances_per_pattern_dict[pat.name] = dists

      # print(pat.name)

    self.distances_per_pattern_dict = distances_per_pattern_dict
    return self.distances_per_pattern_dict

  def print_structured(self, numbered_only=False):
    self.structure.print_structured(self, numbered_only)

  def subdoc(self, start, end):

    assert self.tokens is not None
    #         assert self.embeddings is not None
    #         assert self.distances_per_pattern_dict is not None

    klazz = self.__class__
    sub = klazz("REF")
    sub.start = start
    sub.end = end
    sub.right_padding = 0

    if self.embeddings is not None:
      sub.embeddings = self.embeddings[start:end]

    if self.distances_per_pattern_dict is not None:
      sub.distances_per_pattern_dict = {}
      for d in self.distances_per_pattern_dict:
        sub.distances_per_pattern_dict[d] = self.distances_per_pattern_dict[d][start:end]

    sub.tokens = self.tokens[start:end]
    sub.tokens_cc = self.tokens_cc[start:end]
    return sub

  def normalize_sentences_bounds(self, text):
    """
        splits text into sentences, join sentences with \n
        :param text:
        :return:
        """
    sents = ru_tokenizer.tokenize(text)
    for s in sents:
      s.replace('\n', ' ')

    return '\n'.join(sents)

  def preprocess_text(self, text):
    a = text
    a = normalize_text(a,
                       dates_regex + abbreviation_regex + fixtures_regex +
                       spaces_regex + syntax_regex + numbers_regex +
                       tables_regex)

    # a = self.normalize_sentences_bounds(a)
    return a

  def read(self, name):
    print("reading...", name)
    self.filename = name
    txt = ""
    with open(name, 'r') as f:
      self.set_original_text(f.read())

  def set_original_text(self, txt):
    self.original_text = txt
    self.tokens = None
    self.embeddings = None
    self.normal_text = None

  def tokenize(self, _txt):

    _words = tokenize_text(_txt)

    sparse_words = []
    end = len(_words)
    last_cr_index = 0
    for i in range(end):
      if (_words[i] == '\n') or i == end - 1:
        chunk = _words[last_cr_index:i + 1]
        if (self.right_padding > 0):
          chunk.extend([TEXT_PADDING_SYMBOL] * self.right_padding)
        sparse_words += chunk
        last_cr_index = i + 1

    return sparse_words

  # def parse(self, txt=None):
  #   if txt is None: txt = self.original_text
  #   self.normal_text = self.preprocess_text(txt)
  #
  #   self.tokens = self.tokenize(self.normal_text)
  #   self.tokens_cc = np.array(self.tokens)
  #
  #   return self.tokens

  def parse(self, txt=None):
    if txt is None: txt = self.original_text
    self.normal_text = self.preprocess_text(txt)

    self.structure = DocumentStructure()
    self.tokens, self.tokens_cc = self.structure.detect_document_structure(self.normal_text)

    # self.tokens = self.tokenize(self.normal_text)
    # self.tokens_cc = np.array(self.tokens)

    return self.tokens

  def embedd(self, pattern_factory):
    max_tokens = 6000
    if len(self.tokens) > max_tokens:
      self._embedd_large(pattern_factory.embedder, max_tokens)
    else:
      self.embeddings = self._emb(self.tokens, pattern_factory.embedder)

    print_prof_data()

  @profile
  def _emb(self, tokens, embedder):
    embeddings, _g = embedder.embedd_tokenized_text([tokens], [len(tokens)])
    embeddings = embeddings[0]
    return embeddings

  @profile
  def _embedd_large(self, embedder, max_tokens=6000):

    overlap = int(max_tokens / 5)  # 20%

    number_of_windows = 1 + int(len(self.tokens) / max_tokens)
    window = max_tokens

    print(
      "WARNING: Document is too large for embedding: {} tokens. Splitting into {} windows overlapping with {} tokens ".format(
        len(self.tokens), number_of_windows, overlap))
    start = 0
    embeddings = None
    tokens = []
    while start < len(self.tokens):
      #             start_time = time.time()
      subtokens = self.tokens[start:start + window + overlap]
      print("Embedding region:", start, len(subtokens))

      sub_embeddings = self._emb(subtokens, embedder)

      sub_embeddings = sub_embeddings[0:window]
      subtokens = subtokens[0:window]

      if embeddings is None:
        embeddings = sub_embeddings
      else:
        embeddings = np.concatenate([embeddings, sub_embeddings])
      tokens += subtokens

      start += window
      #             elapsed_time = time.time() - start_time
      #             print ("Elapsed time %d".format(t))
      print_prof_data()

    self.embeddings = embeddings
    self.tokens = tokens


class LegalDocumentLowCase(LegalDocument):

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

  def parse(self, txt=None):
    if txt is None: txt = self.original_text
    self.normal_text = self.preprocess_text(txt)

    self.tokens_cc = self.tokenize(self.normal_text)

    self.normal_text = self.normal_text.lower()

    self.tokens = self.tokenize(self.normal_text)

    return self.tokens


class ContractDocument(LegalDocumentLowCase):
  def __init__(self, original_text):
    LegalDocumentLowCase.__init__(self, original_text)


def rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0):
  c = 0
  sum = None

  for p in distances_per_pattern_dict:
    if p.startswith(prefix):
      x = distances_per_pattern_dict[p]
      if sum is None:
        sum = np.zeros(len(x))

      sum += relu(x, relu_th)
      c += 1
  #   deal/=c
  return sum, c


def mean_by_pattern_prefix(distances_per_pattern_dict, prefix):
  #     print('mean_by_pattern_prefix', prefix, relu_th)
  sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0)
  return normalize(sum)


def rectifyed_normalized_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
  return normalize(rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th))


def rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
  #     print('mean_by_pattern_prefix', prefix, relu_th)
  sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th)
  sum /= c
  return sum


class BasicContractDocument(LegalDocumentLowCase):

  def __init__(self, original_text=None):
    LegalDocumentLowCase.__init__(self, original_text)

  def get_subject_ranges(self, indexes_zipped, section_indexes: List):

    # res = [None] * len(section_indexes)
    # for sec in section_indexes:
    #     for i in range(len(indexes_zipped) - 1):
    #         if indexes_zipped[i][0] == sec:
    #             range1 = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
    #             res[sec] = range1
    #
    #     if res[sec] is None:
    #         print("WARNING: Section #{} not found!".format(sec))
    #
    # return res

    subj_range = None
    head_range = None
    for i in range(len(indexes_zipped) - 1):
      if indexes_zipped[i][0] == 1:
        subj_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
      if indexes_zipped[i][0] == 0:
        head_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
    if head_range is None:
      print("WARNING: Contract type might be not known!!")
      head_range = range(0, 0)
    if subj_range is None:
      print("WARNING: Contract subject might be not known!!")
      if len(self.tokens) < 80:
        _end = len(self.tokens)
      else:
        _end = 80
      subj_range = range(0, _end)
    return head_range, subj_range

  def find_subject_section(self, pattern_fctry: AbstractPatternFactory, numbers_of_patterns):

    self.split_into_sections(pattern_fctry.paragraph_split_pattern)
    indexes_zipped = self.section_indexes

    head_range, subj_range = self.get_subject_ranges(indexes_zipped, [0, 1])

    distances_per_subj_pattern_, ranges_, winning_patterns = pattern_fctry.subject_patterns.calc_exclusive_distances(
      self.embeddings,
      text_right_padding=0)
    distances_per_pattern_t = distances_per_subj_pattern_[:, subj_range.start:subj_range.stop]

    ranges = [np.nanmin(distances_per_subj_pattern_ ),
              np.nanmax(distances_per_subj_pattern_ )]

    weight_per_pat = []
    for row in distances_per_pattern_t:
      weight_per_pat.append(np.nanmin(row))

    print("weight_per_pat", weight_per_pat)

    _ch_r = numbers_of_patterns['charity']
    _co_r = numbers_of_patterns['commerce']

    chariy_slice = weight_per_pat[_ch_r[0]:_ch_r[1]]
    commerce_slice = weight_per_pat[_co_r[0]:_co_r[1]]

    min_charity_index = min_index(chariy_slice)
    min_commerce_index = min_index(commerce_slice)

    print("min_charity_index, min_commerce_index", min_charity_index, min_commerce_index)
    self.per_subject_distances = [
      np.nanmin(chariy_slice),
      np.nanmin(commerce_slice)]

    self.subj_range = subj_range
    self.head_range = head_range

    return ranges, winning_patterns



  #     return

  def analyze(self, pattern_factory):
    self.embedd(pattern_factory)
    self._find_sum(pattern_factory)

    self.subj_ranges, self.winning_subj_patterns = self.find_subject_section(
      pattern_factory, {"charity": [0, 5], "commerce": [5, 5 + 7]})


# SUMS -----------------------------


class ProtocolDocument(LegalDocumentLowCase):

  def __init__(self, original_text=None):
    LegalDocumentLowCase.__init__(self, original_text)

  def make_solutions_mask(self):

    section_name_to_weight_dict = {}
    for i in range(1, 5):
      cap = 'p_cap_solution{}'.format(i)
      section_name_to_weight_dict[cap] = 0.5

    mask = mask_sections(section_name_to_weight_dict, self)
    mask += 0.5
    if self.right_padding > 0:
      mask = mask[0:-self.right_padding]

    mask = smooth(mask, window_len=12)
    return mask

  def find_sum_in_section____(self):
    assert self.subdocs is not None

    sols = {}
    for i in range(1, 5):
      cap = 'p_cap_solution{}'.format(i)

      solution_section = find_section_by_caption(cap, self.subdocs)
      sols[solution_section] = cap

    results = []
    for solution_section in sols:
      cap = sols[solution_section]
      #             print(cap)
      # TODO:
      # render_color_text(solution_section.tokens, solution_section.distances_per_pattern_dict[cap])

      x = extract_sum_from_doc(solution_section)
      results.append(x)

    return results


# Support masking ==================

def find_section_by_caption(cap, subdocs):
  solution_section = None
  mx = 0;
  for subdoc in subdocs:
    d = subdoc.distances_per_pattern_dict[cap]
    _mx = d.max()
    if _mx > mx:
      solution_section = subdoc
      mx = _mx
  return solution_section


def mask_sections(section_name_to_weight_dict, doc):
  mask = np.zeros(len(doc.tokens))

  for name in section_name_to_weight_dict:
    section = find_section_by_caption(name, doc.subdocs)
    #         print([section.start, section.end])
    mask[section.start:section.end] = section_name_to_weight_dict[name]
  return mask


# Charter Docs


class CharterDocument(LegalDocument):
  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)
    self.right_padding = 0

  def tokenize(self, _txt):
    return tokenize_text(_txt)


def max_by_pattern_prefix(distances_per_pattern_dict, prefix, attention_vector=None):
  ret = {}

  for p in distances_per_pattern_dict:
    if p.startswith(prefix):
      x = distances_per_pattern_dict[p]

      if attention_vector is not None:
        x = np.array(x)
        x += attention_vector

      max = x.argmax()
      ret[p] = max

  return ret


def split_into_sections(doc, caption_indexes):
  sorted_keys = sorted(caption_indexes, key=lambda s: caption_indexes[s])

  doc.subdocs = []
  for i in range(1, len(sorted_keys)):
    key = sorted_keys[i - 1]
    next_key = sorted_keys[i]
    start = caption_indexes[key]
    end = caption_indexes[next_key]
    print(key, [start, end])

    subdoc = doc.subdoc(start, end)
    subdoc.filename = key
    doc.subdocs.append(subdoc)


def extract_sum_from_doc(doc: LegalDocument, attention_mask=None, relu_th=0.5):
  sum_pos, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max', relu_th=relu_th)
  sum_neg, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max_neg', relu_th=relu_th)

  sum_pos -= sum_neg

  sum_pos = smooth(sum_pos, window_len=8)
  #     sum_pos = relu(sum_pos, 0.65)

  if attention_mask is not None:
    sum_pos *= attention_mask

  sum_pos = normalize(sum_pos)

  return _extract_sums_from_distances(doc, sum_pos), sum_pos


def _extract_sum_from_distances____(doc: LegalDocument, sums_no_padding):
  max_i = np.argmax(sums_no_padding)
  start, end = get_sentence_bounds_at_index(max_i, doc.tokens)
  sentence_tokens = doc.tokens[start + 1:end]

  f, sentence = extract_sum_from_tokens(sentence_tokens)

  return (f, (start, end), sentence)


def _extract_sums_from_distances(doc: LegalDocument, x):
  maximas = extremums(x)

  results = []
  for max_i in maximas:
    start, end = get_sentence_bounds_at_index(max_i, doc.tokens)
    sentence_tokens = doc.tokens[start + 1:end]

    f, sentence = extract_sum_from_tokens(sentence_tokens)

    if f is not None:
      result = {
        'sum': f,
        'region': (start, end),
        'sentence': sentence,
        'confidence': x[max_i]
      }
      results.append(result)

  return results


MIN_DOC_LEN = 5


def make_soft_attention_vector(doc, pattern_prefix, relu_th=0.5, blur=60, norm=True):
  assert doc.distances_per_pattern_dict is not None

  if len(doc.tokens) < MIN_DOC_LEN:
    print("----ERROR: make_soft_attention_vector: too few tokens {} ".format(untokenize(doc.tokens_cc)))
    return np.full(len(doc.tokens), 0.0001)

  attention_vector, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix,
                                                         relu_th=relu_th)
  attention_vector = relu(attention_vector, relu_th=relu_th)

  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector = smooth_safe(attention_vector, window_len=blur)
  try:
    if norm:
      attention_vector = normalize(attention_vector)
  except:
    print(
      "----ERROR: make_soft_attention_vector: attention_vector for pattern prefix {} is not contrast, len = {}".format(
        pattern_prefix, len(attention_vector)))
    attention_vector = np.full(len(attention_vector), attention_vector[0])

  return attention_vector


def soft_attention_vector(doc, pattern_prefix, relu_th=0.5, blur=60, norm=True):
  assert doc.distances_per_pattern_dict is not None

  if len(doc.tokens) < MIN_DOC_LEN:
    print("----ERROR: soft_attention_vector: too few tokens {} ".format(untokenize(doc.tokens_cc)))
    return np.full(len(doc.tokens), 0.0001)

  attention_vector, c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix, relu_th=relu_th)
  assert c > 0
  attention_vector = relu(attention_vector, relu_th=relu_th)

  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector /= c
  try:
    if norm:
      attention_vector = normalize(attention_vector)
  except:
    print("----ERROR: soft_attention_vector: attention_vector for pattern prefix {} is not contrast, len = {}".format(
      pattern_prefix, len(attention_vector)))

    attention_vector = np.full(len(attention_vector), attention_vector[0])
  return attention_vector


def _find_sentences_by_attention_vector(doc, _attention_vector, relu_th=0.5):
  attention_vector = relu(_attention_vector, relu_th)
  maxes = extremums(attention_vector)[1:]
  maxes = doc.find_sentence_beginnings(maxes)
  maxes = remove_similar_indexes(maxes, 6)

  res = {}
  for i in maxes:
    s, e = get_sentence_bounds_at_index(i + 1, doc.tokens)
    if e - s > 0:
      res[s] = e

  return res, attention_vector, maxes


def embedd_headlines(headline_indexes: List[int], doc: LegalDocument, factory: AbstractPatternFactory, max_len=40) -> \
        List[LegalDocument]:
  _str = doc.structure.structure

  embedded_headlines = []

  tokenized_sentences_list = []
  for i in headline_indexes:
    line:StructureLine = _str[i]

    _len = line.span[1] - line.span[0]
    _len = min(max_len, _len)

    subdoc = doc.subdoc(line.span[0], line.span[0] + _len)
    subdoc.right_padding = 0

    tokenized_sentences_list.append(subdoc.tokens)
    embedded_headlines.append(subdoc)

  sentences_emb, wrds, lens = embedd_tokenized_sentences_list(factory.embedder, tokenized_sentences_list)

  for i in range(len(headline_indexes)):
    embedded_headlines[i].embeddings = sentences_emb[i][0:lens[i]]
    embedded_headlines[i].calculate_distances_per_pattern(factory)

  return embedded_headlines


# # @at_github
# @deprecated
# def _estimate_headline_probability_for_each_line(TCD: LegalDocument):
#
#
#   def number_of_leading_spaces(_tokens):
#     c_ = 0
#     while c_ < len(_tokens) and _tokens[c_] in ['', ' ', '\t', '\n']:
#       c_ += 1
#     return c_
#
#   lines = np.zeros(len(TCD.structure.structure))
#
#   prev_sentence = []
#   prev_value = 0
#
#   _struct = TCD.structure.structure
#   for i in range(len(_struct)):
#     line = _struct[i]
#
#     sentence = TCD.tokens[line.span[0]: line.span[1]]
#     sentence_cc = TCD.tokens_cc[line.span[0]: line.span[1]]
#
#     if len(sentence_cc) > 1:
#       tr = number_of_leading_spaces(sentence)
#       if tr > 0:
#         sentence = sentence[tr:]
#         sentence_cc = sentence_cc[tr:]
#
#     p = headline_probability(sentence, sentence_cc, prev_sentence, prev_value)
#
#     #     if line.level == 0:
#     #       p += 1
#
#     prev_sentence = sentence
#     lines[i] = p
#     prev_value = p
#
#   return lines

# @deprecated
# def highlight_doc_structure(_doc: LegalDocument):
#   print ('-WARNING- highlight_doc_structure is deprecated')
#   p_per_line = _estimate_headline_probability_for_each_line(_doc)
#
#   def local_contrast(x):
#     blur = 2 * int(len(x) / 20.0)
#     blured = smooth_safe(x, window_len=blur, window='hanning') * 0.99
#     delta = relu(x - blured, 0)
#     r = normalize(delta)
#     return r, blured
#
#   max = np.max(p_per_line)
#   result = relu(p_per_line, max / 3.0)
#   contrasted, smoothed = local_contrast(result)
#
#   r = {
#     'line_probability': p_per_line,
#     'line_probability relu': relu(p_per_line),
#     'accents_smooth': smoothed,
#     'result': contrasted
#   }
#
#   return r
