import re
from typing import List

from ml_tools import relu, normalize
from text_tools import tokenize_text, np, untokenize


def _strip_left(tokens):
  for i in range(len(tokens)):
    if tokens[i] != '.' and tokens[i] != ' ' and tokens[i] != '\t':
      return i
  return len(tokens)


def roman_to_arabic(n):
  roman = n.upper()
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
    return None, (0, 0), last_level

  r = roman_might_be(tokens[0])
  if r is not None:
    # Roman number
    return [r], (0, 1), 0

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

      return n, (0, offset), level


    else:
      # Number not found

      # searching for bulletpoints -
      # x = re.search(r'^\s*[\-|•]\s*', tokens[token_index], flags=re.MULTILINE)
      x = re.search(r'^[\-|•]', tokens[token_index], flags=re.MULTILINE)
      if x is not None:
        return [-1], (0, 1), last_level
      else:
        pass

  return None, (0, 0), last_level


class StructureLine():

  def __init__(self, level=0, number=[], bullet=False, span=(0, 0), text_offset=1, line_number=-1) -> None:
    super().__init__()
    self.number = number
    self.level = level
    self.bullet = bullet
    self.span = span
    self.text_offset = text_offset
    self._possible_levels = []
    self.line_number = line_number

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
      values = untokenize(tokens_cc[self.span[0] + self.text_offset: self.span[1]])

    ln = self.line_number
    if line_number is not None:
      ln = line_number
    print('ds> {}\t'.format(ln), offset, number_str, values, suffix)

  def get_numbered(self) -> bool:
    return len(self.number) > 0

  def get_minor_number(self) -> int:
    return self.number[-1]

  numbered = property(get_numbered)
  minor_number = property(get_minor_number)
  parent_number = property(get_parent_number)


class DocumentStructure:

  def __init__(self):
    self.structure = None
    # self._detect_document_structure(text)

  def tokenize(self, _txt):
    return tokenize_text(_txt)

  def detect_document_structure(self, text):
    lines = text.split('\n')

    line_number = 0

    last_level_known = 0

    structure = []

    tokens = []
    tokens_cc = []

    index = 0
    for row in lines:
      line_number += 1

      line_tokens_cc = self.tokenize(row)
      line_tokens = [s.lower() for s in line_tokens_cc]
      tokens_cc += line_tokens_cc
      tokens += line_tokens

      bullet = False

      if len(line_tokens) > 0:
        # not empty
        number, span, _level = get_tokenized_line_number(line_tokens, last_level_known)

        if number is None:
          number = []

        #           if row.upper() == row: #HEADLINE?
        #             _level = -1
        #             last_level_known = _level

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
          line_number=line_number - 1
        )

        structure.append(section_meta)

        index = len(tokens)

    self.tokens_cc = tokens_cc  ## xxx: for debug only: TODO: remove this line

    structure = self.fix_structure(structure)

    self.structure = structure
    return tokens, tokens_cc

  def fix_structure(self, structure, verbose=False):

    numbered = self._get_numbered_lines(structure)

    # for level in range(0, 1):
    #   print('W' * 40, level)
    #   self._fix_numbered_structure(numbered, level=level, verbose=verbose)
    #   self._update_levels(numbered, verbose)

    for a in range(1):
      self._level_non_numbered(structure)
      self._normalize_levels(structure)

      self._fix_top_level_lines(numbered)
      self._update_levels(numbered, verbose)

      self._level_non_numbered(structure)
      self._normalize_levels(structure)

    return structure

  def _fix_top_level_lines(self, structure, start=0, level_delta=0, max_hole=0):
    for allowed_level_delta in [0, 1]:
      for check_parent in [True, False]:
        for max_hole in [0, 0, 1, 2]:
          self._correct_sequence(structure, 0,
                                 level=0,
                                 level_delta=allowed_level_delta,
                                 max_hole=max_hole,
                                 check_parent=check_parent)

  def _correct_sequence(self, structure, start=0, level=0, level_delta=0, max_hole=0, check_parent=True):
    if len(structure) < 1:
      return

    last = -1
    i = start
    while i < len(structure):

      if self._sequence_continues(structure, i, last):
        structure[i].add_possible_level(max(level, structure[i].level))
        if last > 0:
          structure[i].add_possible_level(max(level, structure[last].level))
          structure[last].add_possible_level(max(level, structure[i].level))

        last = i
        i += 1
        pass
      else:
        # Subsequence? or end of sequence?
        if structure[i].minor_number == 1:
          i = self._correct_sequence(structure, start=i, level=level + 1, level_delta=level_delta, max_hole=max_hole,
                                     check_parent=check_parent)
        else:
          return i + 1

    return i
    
  def _find_probable_sequences_on_level(self, structure, level):
    sequences = []
    for i in range(len(structure)):
      if structure[i].level == level:
        sequence = np.zeros(len(structure))
        sequences.append(sequence)

        for allowed_level_delta in [0, 1]:
          for check_parent in [True, False]:
            for max_hole in [0, 0, 1, 2]:
              self._search_possibilities(structure, [i], sequence, 1.1,
                                         check_parent=check_parent,
                                         allowed_level_delta=allowed_level_delta,
                                         max_hole=max_hole)

        return sequences
    return sequences

  def _normalize_levels(self, structure):
    minlevel = structure[0].level
    for s in structure:
      if s.level < minlevel:
        minlevel = s.level
    for s in structure:
      s.level -= minlevel

  def _get_numbered_lines(self, structure):
    numbered = []
    for s in structure:
      if s.numbered:
        # numbered
        numbered.append(s)
    return numbered

  def _update_levels(self, numbered, verbose):
    # DEBUG
    for i in range(len(numbered)):
      line = numbered[i]

      # fixing:
      if len(line.number) < 2:
        line.level = line.get_median_possible_level()
      # print(line.level)
      if verbose:
        line.print(self.tokens_cc, str(line.level) + '--->' + str(line._possible_levels) + ' i:' + str(i))
      # line._possible_levels = []

  def _level_non_numbered(self, structure):
    for s in structure:
      last_level = 1
      if s.numbered:
        last_level = s.level
      else:
        # non numbered
        if s.level < last_level + 1:
          s.level = last_level + 1

  def _sequence_continues(self, structure: List, index: int, index_prev: int, allowed_level_delta=0, check_parent=True,
                          max_hole=0):
    def same_parent(a: StructureLine, b: StructureLine):
      if not check_parent:
        return True

      if len(a.number) > 1:
        if len(b.number) == 1:
          return True  # parent unknown
        elif len(b.number) > 1:
          return b.number[-2] == a.number[-2]

      if len(b.number) > 1:
        if len(a.number) == 1:
          return True  # parent unknown
        elif len(a.number) > 1:
          return b.number[-2] == a.number[-2]

      return True

    if index_prev < 0:
      return True

    curr = structure[index]
    prev = structure[index_prev]

    if 0 <= (curr.minor_number - (prev.minor_number + 1)) <= max_hole and same_parent(curr, prev):
      return abs(curr.level - prev.level) <= allowed_level_delta

    return False





  def find_by_number_on_level(self, start_index, structure, allowed_level_delta=0, check_parent=True, max_hole=0):
    continuations = []

    for i in range(start_index + 1, len(structure)):

      if self._sequence_continues(structure, i, start_index,
                                  allowed_level_delta=allowed_level_delta,
                                  check_parent=check_parent,
                                  max_hole=max_hole):
        continuations.append(i)

    return continuations

    #   for j in range(i+1, len(structure)):
    #     line_after = structure[j]
    #
    #     if line_before.minor_number == line_after.minor_number - 1:
    #       line_before.add_possible_level(line_after.level)
    #       line_after.add_possible_level(line_before.level)
    #
    #     if line_after.minor_number < line_before.minor_number:
    #       line_after.add_possible_level( max (line_after.level, line_before.level+1))
    #
    #     if len(line_after.number)>1:
    #       if line_after.number[-2] == line_before.minor_number:
    #         line_after.add_possible_level(line_before.level+1)

    #

  def _find_tails(self, structure, level=0):
    # if sequence is None:
    print('-', '\t', '\t'.join([str(x) for x in range(len(structure))]))

    if len(structure) < 2:
      return

    sequences = []

    print()
    cnt = 0
    # sequence = np.zeros(len(structure))
    for i in range(len(structure)):
      if structure[i].level == level or True:
        cnt += 1
        sequence = np.zeros(len(structure))
        # lev = structure[i].level
        self._search_possibilities(structure, [i], sequence, 2.1, check_parent=True, allowed_level_delta=0)
        self.__uplevel_sequence_holes(sequence, structure, level)

        # self._search_possibilities(structure, [i], sequence, 1.2, check_parent=False, allowed_level_delta=0)
        # self.__process_sequence(sequence, structure, level)
        # self._search_possibilities(structure, [i], sequence, 1.1, check_parent=True, allowed_level_delta=1)
        # self.__process_sequence(sequence, structure, level)
        # self._search_possibilities(structure, [i], sequence, 1, check_parent=False, allowed_level_delta=1)
        # self.__process_sequence(sequence, structure, level)
        # self._search_possibilities(structure, [i], sequence, 1, check_parent=True, max_hole=1)  # allowing 2 elements missing in sqeq

        # sequence  = relu(sequence,  0)

        sequences.append(sequence)
        print(i, '\t', '\t'.join([str(int(x)) for x in sequence]), '\t\t', 'x')

        # print ('sequence=',sequence)

    # sequence = np.sum(sequences,0)
    #
    # sequence = normalize(sequence)
    # sequence *= 10
    # sequence = relu(sequence, 5)
    # print('x', '\t', '\t'.join([str(int(x)) for x in sequence]), '\t\t', 'x')

    if cnt == 0:
      self._find_tails(structure, level + 1)

    # xx= normalize(np.mean(sequences,0))
    # print(xx)

    # for col in range(len(structure)):
    #   # ss_col = StructureLine()
    #   for row in range(len(structure)):
    #     sequence=sequences[row]
    #     probabilty = sequence[col]
    #     for x in range(int(probabilty)):
    #       # ss_col.add_possible_level(structure[row].level)
    #       structure[col].add_possible_level(structure[row].level)
    #       # structure[row].add_possible_level(structure[col].level)
    #
    #   # _mean_level = ss_col.get_median_possible_level()
    #   # structure[col].add_possible_level(_mean_level)
    #   print(col, '\t', '\t'.join([str(int(x)) for x in sequences[col]]), '\t\t', 'x')

    # for i in range(len(structure)):
    #   sequence = relu(sequences[i], 0)
    #
    #   ss = StructureLine()
    #   for j in range(len(sequence)):
    #     probabilty = sequence[j]
    #     for x in range(int(probabilty)):
    #       ss.add_possible_level(structure[j].level)
    #
    #   _mean_level = ss.get_median_possible_level()
    #   # print(ss.get_median_possible_level(),ss._possible_levels)
    #
    #   print(i, '\t', '\t'.join([str(int(x)) for x in sequence]), '\t\t', _mean_level)
    #   structure[i].add_possible_level(_mean_level)

  def __uplevel_sequence_holes(self, sequence, structure, min_level):

    try:
      sequence_norm = normalize(sequence)
      sequence_norm *= 10

      # get non-zero indexex
      indices = np.nonzero(relu(sequence_norm, 1))[0]
      # process sequences between points
      self._uplevel_lines_between_indexes(structure, indices, min_level)

      # get non-zero indexex
      indices = np.nonzero(relu(sequence_norm, 2))[0]
      # process sequences between points
      self._uplevel_lines_between_indexes(structure, indices, min_level)

      # get non-zero indexex
      indices = np.nonzero(relu(sequence_norm, 3))[0]
      # process sequences between points
      self._uplevel_lines_between_indexes(structure, indices, min_level)

      # indices = np.nonzero(relu(sequence_norm, 7))[0]
      # # process sequences between points
      # self._uplevel_lines_between_indexes(structure, indices, min_level)


    except:
      return

  def _uplevel_lines_between_indexes(self, structure, indices, min_level):
    def set_min_level(struct, min_level):
      for subline in struct:
        subline.add_possible_level(max(min_level, subline.level))

    prev = 0
    for k in indices:
      structure[k].add_possible_level(min_level - 1)

      substr = structure[prev:k]
      set_min_level(substr, min_level)
      self._fix_top_level_lines(substr, min_level)
      prev = k + 1
    # last segement
    substr = structure[prev:]
    set_min_level(substr, min_level)

  def _search_possibilities(self, structure, indexes, sequence, probability, check_parent=True, allowed_level_delta=0,
                            max_hole=0):

    for c_index in indexes:
      sequence[c_index] += probability

      possible_nexts = self.find_by_number_on_level(c_index, structure,
                                                    allowed_level_delta=allowed_level_delta,
                                                    check_parent=check_parent,
                                                    max_hole=max_hole)
      self._search_possibilities(structure, possible_nexts, sequence, probability,
                                 allowed_level_delta=allowed_level_delta,
                                 check_parent=check_parent,
                                 max_hole=max_hole)



  def print_structured(self, doc, numbered_only=False):
    ln = 0
    for s in self.structure:
      if s.numbered or not numbered_only:
        s.print(doc.tokens_cc, str(s.level) + '->' + str(s._possible_levels), line_number=ln)
        ln += 1
