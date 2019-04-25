import re

from ml_tools import *
from text_tools import tokenize_text, np, untokenize


def _strip_left(tokens):
  for i in range(len(tokens)):
    if tokens[i] != '.' and tokens[i] != ' ' and tokens[i] != '\t':
      return i
  return len(tokens)


def roman_to_arabic(n):
  roman = n.upper().lstrip()
  if not check_valid_roman(roman):
    return None

  keys = ['IV', 'IX', 'XL', 'XC', 'CD', 'CM', 'I', 'V', 'X', 'L', 'C', 'D', 'M']
  to_arabic = {'IV': '4', 'IX': '9', 'XL': '40', 'XC': '90', 'CD': '400', 'CM': '900',
               'I': '1', 'V': '5', 'X': '10', 'L': '50', 'C': '100', 'D': '500', 'M': '1000'}
  for key in keys:
    if key in roman:
      roman = roman.replace(key, ' {}'.format(to_arabic.get(key)))
  return sum(int(num) for num in roman.split())


def check_valid_roman(roman):
  if len(roman.strip()) == 0:
    return False
  invalid = ['IIII', 'VV', 'XXXX', 'LL', 'CCCC', 'DD', 'MMMM', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  if any(sub in roman for sub in invalid):
    return False
  return True


def roman_might_be(wrd):
  try:
    return roman_to_arabic(wrd)
  except:
    return None


def string_to_ip(str):
  ret = []
  n = str.split('.')
  for c in n:
    try:
      ret.append(int(c))
    except:
      pass
  return ret


def get_tokenized_line_number(tokens: List, last_level):
  """
  :param tokens: list of stings, supposed to be lowercase
  :return: number, number_region, line???, level (for 1.1.1 it is 3)
  """

  if len(tokens) == 0:
    return None, (0, 0), last_level + 1, False

  r = roman_might_be(tokens[0])
  if r is not None:
    # Roman number
    return [r], (0, 1), 0, True

  else:
    # Arabic (might be)
    # Searching for numbered lines
    token_index = 0

    if tokens[0] == 'статья' and len(tokens) >= 2:
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

        offset += _strip_left(tokens[offset:])

      return n, (0, offset), level, False


    else:
      # Number not found

      # searching for bulletpoints -
      # x = re.search(r'^\s*[\-|•]\s*', tokens[token_index], flags=re.MULTILINE)
      x = re.search(r'^[\-|•]', tokens[token_index], flags=re.MULTILINE)
      if x is not None:
        return [-1], (0, 1), last_level, False
      else:
        pass

  return None, (0, 0), last_level, False


class StructureLine():

  def __init__(self, level=0, number=[], bullet=False, span=(0, 0), text_offset=1, line_number=-1, roman=False) -> None:
    super().__init__()
    self.number = number
    self.level = level
    self.bullet = bullet
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

  def get_parent_number(self):
    if len(self.number) > 1:
      return self.number[-2]

    return None

  def add_possible_level(self, l):
    self._possible_levels.append(l)

  def print(self, tokens_cc, suffix='', line_number=None):

    offset = '  .  ' * self.level

    number_str = '.'.join([str(x) for x in self.number])
    if self.bullet:
      number_str = '• '
    if self.numbered:
      number_str += '.'
    #         print(offset, number_str, (self.tokens_cc[span[0] + number_tokens:span[1]]))
    values = "not text so far"
    if tokens_cc is not None:
      values = self.to_string_no_number(tokens_cc)

    ln = self.line_number
    if line_number is not None:
      ln = line_number

    se = '+' * self.sequence_end
    #     if self.sequence_end>0:
    #       se=str(self.sequence_end)
    print('ds>{}\t {}\t'.format(ln, se), offset, number_str, values, suffix)

  def to_string_no_number(self, tokens_cc):
    return untokenize(tokens_cc[self.span[0] + self.text_offset: self.span[1]])

  def to_string(self, tokens):
    return untokenize(tokens[self.slice])

  def subtokens(self, tokens):
    return tokens[self.span[0]: self.span[1]]

  def get_numbered(self) -> bool:
    return len(self.number) > 0

  def get_minor_number(self) -> int:
    if (self.numbered):
      return self.number[-1]

  numbered = property(get_numbered)
  minor_number = property(get_minor_number)
  parent_number = property(get_parent_number)


class DocumentStructure:

  def __init__(self):
    self.structure: List[StructureLine] = None
    self.headline_indexes = []
    # self._detect_document_structure(text)

  def tokenize(self, _txt):
    return tokenize_text(_txt)

  def detect_document_structure(self, text):
    lines: List[str] = text.split('\n')

    last_level_known = 0

    structure = []

    tokens = []
    tokens_cc = []

    index = 0
    romans = 0
    maxroman = 0
    for __row in lines:

      line_tokens_cc = self.tokenize(__row.strip()) + ['\n']

      line_tokens = [s.lower() for s in line_tokens_cc]
      tokens_cc += line_tokens_cc
      tokens += line_tokens

      bullet = False

      if len(line_tokens) > 0:
        # not empty
        number, span, _level, roman = get_tokenized_line_number(line_tokens, last_level_known)

        if roman: romans += 1

        if number is None:
          number = []

        else:
          last_level_known = _level
          if number[-1] < 0:
            bullet = True
            number = []

        # level , section number, is it a bullet, span : (start, end)
        section_meta = StructureLine(
          level=_level,  # 0
          number=number,  # 1
          bullet=bullet,  # 3
          span=(index, index + len(line_tokens)),
          text_offset=span[1],
          line_number=len(structure),
          roman=roman
        )

        structure.append(section_meta)
        index = len(tokens)

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

    self.headline_indexes = self._find_headlines(tokens, tokens_cc)
    self.headline_indexes = self._merge_headlines_if_underlying_section_is_tiny(self.headline_indexes)
    return tokens, tokens_cc



  #     del _charter_doc.structure.structure[i]

  def next_headline_after(self, start:int) -> int:
    for si in self.headline_indexes:
      line_start = self.structure[si].span[0]
      if line_start > start:
        return line_start

    return self.structure[-1].span[-1] #end of the doc

  def _merge_headlines_if_underlying_section_is_tiny(self, headline_indexes) -> List[int]:
    indexes_to_remove = []

    slines = self.structure

    line_i = 0
    while line_i < len(headline_indexes) - 1:
      i_this = headline_indexes[line_i]
      i_next = headline_indexes[line_i + 1]

      sline: StructureLine = slines[i_this]
      sline_next: StructureLine = slines[i_next]

      section_size = sline_next.span[0] - sline.span[1]

      if section_size < 20:
        self._merge_headlines(headline_indexes, line_i, line_i + 1, indexes_to_remove)
      else:
        line_i += 1

    return headline_indexes

  def _merge_headlines(self, headline_indexes, line_i, line_i_next, indexes_to_remove):

    i_this =  headline_indexes[line_i]
    i_next =  headline_indexes[line_i_next]

    sline: StructureLine = self.structure[i_this]
    sline_next: StructureLine = self.structure[i_next]


    sline.span = (sline.span[0], sline_next.span[1])

    del headline_indexes[line_i_next]
    for i in range(i_this + 1, i_next + 1):
      indexes_to_remove.append(i)

  def _find_headlines(self, tokens, tokens_cc) -> List[int]:

    headlines_probability = np.zeros(len(self.structure))

    prev_sentence = []
    prev_value = 0
    for i in range(len(self.structure)):
      line = self.structure[i]
      line_tokens = line.subtokens(tokens)
      line_tokens_cc = line.subtokens(tokens_cc)

      p = headline_probability(line_tokens, line_tokens_cc, line, prev_sentence, prev_value)
      headlines_probability[i] = p
      line.add_possible_level(p)
      prev_sentence = line_tokens
      prev_value = p

    """ 🧠🕺 magic an brainfu** inside """
    _contrasted_probability = self._highlight_headlines_probability(headlines_probability)
    headline_indexes = sorted(np.nonzero(_contrasted_probability)[0])

    return remove_similar_indexes_considering_weights(headline_indexes, _contrasted_probability)

  def _highlight_headlines_probability(self, p_per_line):

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

    numbered = self._get_numbered_lines(structure)
    if len(numbered) == 0:
      return structure

    # for a in range(1):
    # self._uplevel_non_numbered(structure)
    self._normalize_levels(structure)

    # self._fix_top_level_lines(numbered)
    # self._update_levels(structure, verbose)

    # self._uplevel_non_numbered(structure)
    # self._normalize_levels(structure)

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

  def _get_numbered_lines(self, structure):
    numbered = []
    for s in structure:
      if s.numbered:
        # numbered
        numbered.append(s)
    return numbered

  def _update_levels(self, seq, verbose):
    # DEBUG
    for i in range(len(seq)):
      line = seq[i]

      # fixing:
      if len(line.number) < 2:
        line.level = line.get_median_possible_level()

      # if verbose:
      #   line.print(self.tokens_cc, str(line.level) + '--->' + str(line._possible_levels) + ' i:' + str(i))

  def _uplevel_non_numbered(self, structure: List[StructureLine]):
    for s in structure:
      _last_level = 1
      if s.numbered:
        _last_level = s.level
      else:
        # non numbered
        if len(s._possible_levels) > 0:
          s.level = s.get_median_possible_level()
        elif s.level < _last_level + 1:
          s.level = _last_level + 1

  def print_structured(self, doc, numbered_only=False):
    ln = 0
    for s in self.structure:
      if s.numbered or not numbered_only:
        s.print(doc.tokens_cc, str(s.level) + '->' + str(s._possible_levels), line_number=ln)
        ln += 1


# ---------------
strange_symbols = re.compile(r'[_$@+]–')


def headline_probability(sentence: List[str], sentence_cc, sentence_meta: StructureLine, prev_sentence,
                         prev_value) -> float:
  """
  _cc == original case
  """

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

  # headline may not go after another headline
  if prev_value > 0:
    value -= prev_value / 2

  # if it ends with a number, it is a contents-line
  if len(sentence) > 3:
    r_off = 2
    if sentence[-r_off] == '.':
      r_off = 3

    if sentence[-r_off].isdigit():
      value -= 1.8

  # span = sentence_meta.span
  _level = sentence_meta.level
  # number, span, _level = get_tokenized_line_number(sentence, None)
  row = untokenize(sentence_cc[sentence_meta.text_offset:])[:40]
  row = row.lstrip()

  if strange_symbols.search(row) is not None:
    value -= 2

  if sentence_meta.numbered:

    # headline starts from 'статья'
    if sentence[0] == 'статья':
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

  # ------- any number
  # headline DOES not start from lowercase
  if len(row) > 0:
    if row.lower()[0] == row[0]:
      value -= 3

  # headline is UPPERCASE
  if row.upper() == row:
    if not row.isdigit(): #there some trash
      value += 1.5

  if prev_sentence == ['\n'] and sentence != ['\n']:
    value += 1

  return value

#XXXL
def remove_similar_indexes_considering_weights(indexes: List[int], weights: List[float]) -> List[int]:
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
