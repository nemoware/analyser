import re

from ml_tools import *
from text_tools import *


def _count_start_whitespace(tokens: Tokens) -> int:
  for i in range(len(tokens)):
    if tokens[i] != '.' and tokens[i] != ' ' and tokens[i] != '\t':
      return i
  return len(tokens)


def get_tokenized_line_number(tokens: Tokens, last_level):
  """	
  :param tokens: list of stings, supposed to be lowercase	
  :return: number, number_region, line???, level (for 1.1.1 it is 3)	
  """

  if len(tokens) == 0:
    return None, (0, 0), last_level + 1, False

  r = roman_might_be(tokens[0])

  if r:
    # Roman number
    return [r], (0, 1), 0, True

  else:
    # Arabic (might be)
    # Searching for numbered lines
    token_index = 0

    if tokens[0].lower() == 'статья' and len(tokens) >= 2:
      token_index = 1

    x = re.search(r'^(\d{1,2}($|\.)){1,6}', tokens[token_index], flags=re.MULTILINE)
    if x is not None:
      # found number
      n = string_to_ip(x.group(0))
      level = len(n)
      offset = token_index + 1  # tokens before actual text

      if len(tokens) > 2:
        if tokens[token_index + 1] == ')':
          level += 1
          offset += 1

        offset += _count_start_whitespace(tokens[offset:])

      return n, (0, offset), level, False

    else:
      # Number not found
      # searching for bulletpoints -
      x = re.search(r'^[\-|•]', tokens[token_index], flags=re.MULTILINE)
      if x is not None:
        return [-1], (0, 1), last_level, False
      else:
        pass

  return None, (0, 0), last_level, False


class StructureLine():

  def __init__(self, level=0, number=[], bullet: bool = False, span=(0, 0), text_offset=1, line_number=-1,
               roman=False) -> None:
    super().__init__()
    self.number = number
    self.level = level
    self.bullet: bool = bullet
    # @deprecated
    self.span = span
    self.slice = slice(span[0], span[1])
    self.text_offset = text_offset
    self._possible_levels = []
    self.line_number = line_number
    self.sequence_end = 0
    self.roman = roman

  def __str__(self) -> str:
    return ('#{}  N:{}  L:{} -> PL:{}, '.format(self.minor_number, self.number, self.level, self._possible_levels))

  def get_median_possible_level(self):
    if len(self._possible_levels) == 0:
      return self.level

    counts = np.bincount(self._possible_levels)
    return np.argmax(counts)

  def _get_parent_number(self):
    if len(self.number) > 1:
      return self.number[-2]

    return None

  def add_possible_level(self, l):
    self._possible_levels.append(l)

  def _is_numbered(self) -> bool:
    return len(self.number) > 0

  def _get_minor_number(self) -> int:
    if self.numbered:
      return self.number[-1]

  numbered = property(_is_numbered)
  minor_number = property(_get_minor_number)
  parent_number = property(_get_parent_number)


def headline_probability(tokens_map: TextMap, sentence_meta: StructureLine, prev_sentence, prev_value) -> float:
  """	
  _cc == original case	
  """

  sentence: Tokens = tokens_map[sentence_meta.span[0]: sentence_meta.span[1]]

  NEG = -1
  value = 0

  if sentence == ['\n']:
    return NEG

  if len(sentence) < 2:
    return NEG

  if len(sentence) > 20:
    return NEG

  if len(sentence) > 10:
    value -= 2

  # headline is short enough
  if len(sentence) < 10:
    value += 1

  if 3 <= len(sentence) <= 6:
    value += 1

  # headline may not appear right  after another headline
  if prev_value > 0:
    value -= prev_value / 2

  # if it ends with a number, it is a contents-line
  if len(sentence) > 3:
    r_off = 2
    if sentence[-r_off] == '.':
      r_off = 3

    if sentence[-r_off].isdigit():
      value -= 1.8

  _level = sentence_meta.level

  row = tokens_map.text_range(sentence_meta.span)
  row = row.lstrip()

  if strange_symbols.search(row) is not None:
    value -= 2

  if sentence_meta.numbered:

    if len(sentence) < 4:
      return NEG

    # headline starts from 'статья'
    if sentence[0].lower() == 'статья':
      value += 3

    if sentence_meta.minor_number > 0:
      value += 1

    # headline number is NOT too big
    if sentence_meta.minor_number > 40:
      value -= 1

    # headline is NOT a bullet
    if sentence_meta.minor_number < 0:
      return NEG
    # ----
    if _level is not None:
      if _level == 0:
        value += 1

      if _level > 1:
        # headline is NOT a 1.2 - like-numbered
        return -_level

  #   # ------- any number
  else:
    # headline normally DOES not start from lowercase
    if len(row) > 0:
      if row.lower()[0] == row[0]:
        value -= 1

  # headline is UPPERCASE
  if row.upper() == row:
    if not row.isdigit():  # there some trash
      value += 1.5

  if is_blank(''.join(prev_sentence)) and not is_blank(''.join(sentence)):
    value += 1

  return value


class DocumentStructure:

  def __init__(self, headline_probability_f=headline_probability):
    self.structure: List[StructureLine] = None
    self.headline_indexes: List[int] = []
    self.headline_probability_func = headline_probability_f  # make it pluggable

  def detect_document_structure(self, tokens_map: TextMap):
    assert tokens_map is not None

    last_level_known = 0
    structure = []
    romans = 0
    lines_ranges = tokens_map.split_spans('\n', add_delimiter=True)

    for line_span in lines_ranges:

      line_tokens = tokens_map.tokens_by_range(line_span)

      bullet = False

      if line_tokens:
        # not empty line
        number, text_span, _level, roman = get_tokenized_line_number(line_tokens, last_level_known)

        if roman:
          romans += 1

        if not number:
          number = []
        else:
          last_level_known = _level
          if number[-1] < 0:
            bullet = True
            number = []

        # level , section number, is it a bullet, span : (start, end)
        __section_meta = StructureLine(
          level=_level,  # 0
          number=number,  # 1
          bullet=bullet,  # 3
          span=(line_span[0], line_span[1]),
          text_offset=text_span[1],
          line_number=len(structure),
          roman=roman
        )

        structure.append(__section_meta)

      if romans < 3:
        # not enough roman numbers, so these are not roman
        for s in structure:
          if s.roman:
            s.number = []
            s.level = max(2, s.level + 1)

      elif romans > 40:
        # too many, does not look like top-level
        for s in structure:
          if s.roman:
            s.level = max(2, s.level)

    self.structure = self._fix_structure(structure)

    self.headline_indexes = self._find_headlines(tokens_map)
    self.headline_indexes = self._merge_headlines_if_underlying_section_is_tiny(self.headline_indexes)

  #     del _charter_doc.structure.structure[i]

  def next_headline_after(self, start: int) -> int:
    for si in self.headline_indexes:
      line_start = self.structure[si].span[0]
      if line_start > start:
        return line_start

    return self.structure[-1].span[-1]  # end of the doc

  def _merge_headlines_if_underlying_section_is_tiny(self, headline_indexes: [int], min_section_size=15) -> List[int]:
    indexes_to_remove = []

    slines = self.structure

    line_i = 0
    while line_i < len(headline_indexes) - 1:
      i_this = headline_indexes[line_i]
      i_next = headline_indexes[line_i + 1]

      sline: StructureLine = slines[i_this]
      sline_next: StructureLine = slines[i_next]

      section_size = sline_next.span[0] - sline.span[1]

      if section_size < min_section_size:
        self._merge_headlines(headline_indexes, line_i, line_i + 1, indexes_to_remove)
      else:
        line_i += 1

    return headline_indexes

  def _merge_headlines(self, headline_indexes, line_i, line_i_next, indexes_to_remove):

    i_this = headline_indexes[line_i]
    i_next = headline_indexes[line_i_next]

    sline: StructureLine = self.structure[i_this]
    sline_next: StructureLine = self.structure[i_next]

    sline.span = (sline.span[0], sline_next.span[1])

    del headline_indexes[line_i_next]
    for i in range(i_this + 1, i_next + 1):
      indexes_to_remove.append(i)

  def _find_headlines(self, tokens_map: TextMap) -> List[int]:

    headlines_probability = np.zeros(len(self.structure))

    prev_sentence: Tokens = []
    prev_value = 0
    for i in range(len(self.structure)):
      line = self.structure[i]

      p = self.headline_probability_func(tokens_map, line, prev_sentence, prev_value)

      headlines_probability[i] = p
      line.add_possible_level(p)
      prev_sentence = tokens_map.tokens_by_range(line.span)
      prev_value = p

    """ 🧠🕺 Magic an Brainfu** inside """
    _contrasted_probability = self._highlight_headlines_probability(headlines_probability)
    headline_indexes = sorted(np.nonzero(_contrasted_probability)[0])

    return remove_similar_indexes_considering_weights(headline_indexes, _contrasted_probability)

  def _highlight_headlines_probability(self, p_per_line: np.ndarray):
    # TODO: get rid of these magic numbers
    def local_contrast(x):
      blur = 2 * int(len(x) / 20.0)
      blured = smooth_safe(x, window_len=blur, window='hanning') * 0.99
      delta = relu(x - blured, 0)
      return delta, blured

    max = np.max(p_per_line)
    result = relu(p_per_line, max / 3.0)
    contrasted, smoothed = local_contrast(result)

    return contrasted

  def _fix_structure(self, structure, verbose=False):

    numbered = self.get_numbered_lines(structure)
    if len(numbered) == 0:
      return structure

    self._normalize_levels(structure)

    return structure

  def _find_min_level(self, structure):
    min_level = structure[0].level
    for s in structure:
      if s.level < min_level:
        min_level = s.level
    return min_level

  def _normalize_levels(self, structure):
    minlevel = self._find_min_level(structure)

    for s in structure:
      s.level -= minlevel

  def get_numbered_lines(self, structure=None) -> List[StructureLine]:
    if not structure:
      structure = self.structure

    numbered: List[StructureLine] = []
    for s in structure:
      if s.numbered:
        # numbered
        numbered.append(s)
    return numbered

  # def print_structured(self, doc, numbered_only=False):
  #   ln = 0
  #   for s in self.structure:
  #     if s.numbered or not numbered_only:
  #       s.print(doc.tokens_cc, str(s.level) + '->' + str(s._possible_levels), line_number=ln)
  #       ln += 1


# ---------------
strange_symbols = re.compile(r'[_$@+]–')


def is_blank(s: str):
  return s.strip() == ''


# XXXL
def remove_similar_indexes_considering_weights(indexes: List[int], weights: FixedVector) -> List[int]:
  hif = []

  def is_index_far(i):
    if i == 0: return True
    return indexes[i] - indexes[i - 1] > 1

  def is_bigger_confidence(i):
    id = indexes[i]
    id_p = hif[-1]
    return weights[id] > weights[id_p]

  for i in range(len(indexes)):
    id = indexes[i]

    if is_index_far(i):
      hif.append(id)
    elif is_bigger_confidence(i):
      # replace
      hif[-1] = id

  return hif
