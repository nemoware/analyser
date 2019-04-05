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
        return None, (0, 1), last_level
      else:
        pass

  return None, (0, 0), last_level


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

    min_level = None

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
        else:
          last_level_known = _level

        if min_level is None or _level < min_level:
          min_level = _level

        # level , section number, is it a bullet, span : (start, end)
        section_meta = {
          'level': _level,  # 0
          'number': number,  # 1
          'bullet': bullet,  # 3
          'span': (index, index + len(line_tokens)),
          'text_offset': span[1]
        }

        structure.append(section_meta)

        index = len(tokens)

    for s in structure:
      s['level'] -= min_level

    self.tokens_cc = tokens_cc  ## xxx: for debug only: TODO: remove this line

    structure = self.fix_structure(structure)

    self.structure = structure
    return tokens, tokens_cc

  def fix_structure(self, structure):
    numbered = []
    for s in structure:
      if len(s['number']) > 0:
        # numbered
        numbered.append(s)
        s['PL'] = []
      else:
        if s['level'] == 0:
          s['level'] == 2

    for level in range(0, 4):
      print('W' * 40, level)
      self.fix_numbered_structure(numbered, level)

    # DEBUG
    for i in range(len(numbered)):
      line = numbered[i]
      print(line['PL'])
      if len(line['PL']) > 0:
        # fixing:
        line['level'] = int(np.median(line['PL']))
        print(line['level'])
      self._print_line(line, str(line['level']) + '--->' + str(line['PL']) + ' i:' + str(i))

    for s in structure:

      last_level = None
      if len(s['number']) > 0:
        last_level = s['level']
      else:
        # non numbered
        if last_level is not None:
          if s['level'] < last_level + 1:
            s['level'] = last_level + 1

    return structure

  def _fix_numbered_structure(self, structure, level=0):
    def sequence_continues(index, last_index_of_seq, ignore_level=False):
      if last_index_of_seq < 0:
        return True

      prev = structure[last_index_of_seq]
      curr = structure[index]
      if curr['number'][-1] - 1 == prev['number'][-1]:
        if ignore_level:
          return True
        else:
          return curr['level'] == prev['level']

      return False

    def process_substructure(s, e):
      substructure = structure[s:e]
      if len(substructure) > 0:
        self._fix_numbered_structure(substructure, level + 1)

    sequence_on_level = []
    last_index_of_seq = -1
    for i in range(len(structure)):
      line = structure[i]
      if line['level'] == level:
        if sequence_continues(i, last_index_of_seq):
          sequence_on_level.append(i)
          last_index_of_seq = i
        else:
          line['PL'].append(level + 1)

    print('sequence_on_level', level, sequence_on_level, [structure[x]['number'] for x in sequence_on_level])
    # ----------------
    # first segement:
    if len(sequence_on_level) > 0:
      process_substructure(0, sequence_on_level[0])

      # sequence_on_level.append(None)  # end
      for j in range(1, len(sequence_on_level)):
        process_substructure(sequence_on_level[j - 1] + 1, sequence_on_level[j])

      # last segement:
      process_substructure(sequence_on_level[-1], None)

  def print_structured(self, doc):
    for s in self.structure:
      self._print_line(s, doc.tokens_cc)

  def _print_line(self, s, tokens_cc, suffix=''):
    span = s['span']
    number = s['number']
    offset = ' . ' * s['level']
    number_tokens = s['text_offset']
    number_str = '.'.join([str(x) for x in number])
    if s['bullet']:  # Not a blullet
      number_str = '• '
    if len(number) > 0:
      number_str += '.'
    #         print(offset, number_str, (self.tokens_cc[span[0] + number_tokens:span[1]]))
    values = "not text so far"
    if tokens_cc is not None:
      values = untokenize(tokens_cc[span[0] + number_tokens:span[1]])
    print(offset, number_str, values, suffix)
