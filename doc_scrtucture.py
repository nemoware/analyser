import re
from typing import List

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

  def add_possible_level(self, l):
    self._possible_levels.append(l)

  def print(self, tokens_cc, suffix=''):

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

    print('ds> {}\t'.format(self.line_number), offset, number_str, values, suffix)

  def get_numbered(self) -> bool:
    return len(self.number) > 0

  def get_minor_number(self) -> int:
    return self.number[-1]

  numbered = property(get_numbered)
  minor_number = property(get_minor_number)


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

    for s in numbered:
      s.add_possible_level(s.level)

    for s in numbered:
      if (len(s.number) >= 2):
        s.add_possible_level(s.level)

    for level in range(0, 3):
      print('W' * 40, level)
      self._fix_numbered_structure(numbered, level, verbose)

    self._update_levels(numbered, verbose)
    self._level_non_numbered(structure)

    minlevel = structure[0].level
    for s in structure:
      if s.level < minlevel:
        minlevel = s.level

    for s in structure:
      s.level -= minlevel

    return structure

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
      line.level = line.get_median_possible_level()
      # print(line.level)
      if verbose:
        line.print(self.tokens_cc, str(line.level) + '--->' + str(line._possible_levels) + ' i:' + str(i))

  def _level_non_numbered(self, structure):
    for s in structure:
      last_level = 1
      if s.numbered:
        last_level = s.level
      else:
        # non numbered
        if s.level < last_level + 1:
          s.level = last_level + 1

  def _sequence_continues(self, structure, index, index_prev, ignore_level=False):
    if index_prev < 0:
      return True

    curr = structure[index]
    prev = structure[index_prev]
    if curr.minor_number - 1 == prev.minor_number:
      if ignore_level:
        return True
      else:
        return curr.level == prev.level

    return False

  def _fix_numbered_structure(self, structure, level=0, verbose=False):

    def process_substructure(s, e):
      substructure = structure[s:e]
      if len(substructure) > 0:
        self._fix_numbered_structure(substructure, level + 1)

    sequence_on_level = []
    last_index_of_seq = -1

    i = 0
    while i < len(structure):
      # for i in range(len(structure)):
      line = structure[i]

      if line.level == level:

        if self._sequence_continues(structure, i, last_index_of_seq):
          sequence_on_level.append(i)
          last_index_of_seq = i

          i += 1
        else:
          # find_continuation:

          cont_index_min = self.find_continuation(structure[i:],
                                                  structure[last_index_of_seq].minor_number + 1,
                                                  structure[last_index_of_seq].level,
                                                  ignore_level=False)

          cont_index_max = self.find_continuation(structure[i:],
                                                  structure[last_index_of_seq].minor_number + 1,
                                                  structure[last_index_of_seq].level,
                                                  ignore_level=True)

          if cont_index_max > 0:
            for k in range(i, i + cont_index_max):
              structure[k].add_possible_level(max(structure[last_index_of_seq].level + 1, structure[k].level))

          if cont_index_min > 0:
            for k in range(i, i + cont_index_min):
              structure[k].add_possible_level(max(structure[last_index_of_seq].level + 1, structure[k].level))

            sequence_on_level.append(last_index_of_seq)

            i += cont_index_min
            # sequence_on_level.append(i)
            # last_in_sequence = structure[i]
            # i+=1
            if verbose:
              print('sequence_on_level=', sequence_on_level)
          else:
            # TODO: add all
            line.add_possible_level(max(level + 1, line.level))
            i += 1
      else:
        i += 1

    if verbose:
      print('sequence_on_level', level, sequence_on_level, [structure[x].number for x in sequence_on_level])
    # ----------------
    # first segement:
    if len(sequence_on_level) > 0:
      process_substructure(0, sequence_on_level[0])

      # sequence_on_level.append(None)  # end
      for j in range(1, len(sequence_on_level)):
        process_substructure(sequence_on_level[j - 1] + 1, sequence_on_level[j])

      # last segement:
      process_substructure(sequence_on_level[-1], None)

  def find_continuation(self, sub_str, search_for_number, level, ignore_level):

    prev = -1
    for j in range(len(sub_str)):
      line = sub_str[j]

      if self._sequence_continues(sub_str, j, prev, ignore_level=ignore_level):
        prev = j

        if j == len(sub_str) - 1:
          return j
      else:
        if search_for_number == line.minor_number and line.level == level:
          # TODO: check  level also
          return j
        else:
          cnt = self.find_continuation(sub_str[j:], line.minor_number + 1, line.level, ignore_level)
          if (cnt > 0):
            return j + cnt

    return -1

  def print_structured(self, doc, numbered_only=False):
    for s in self.structure:
      if s.numbered or not numbered_only:
        s.print(doc.tokens_cc, str(s._possible_levels))
