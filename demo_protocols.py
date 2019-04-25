# -*- coding: utf-8 -*-

from legal_docs import BasicContractDocument, LegalDocument, deprecated
from parsing import ParsingContext
from patterns import *
from renderer import AbstractRenderer
from transaction_values import extract_sum_from_tokens


class ProtocolRenderer(AbstractRenderer):
  pass


# from split import *
# ----------------------
class ProtocolDocument(BasicContractDocument):
  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

  def _find_sum(self, pattern_factory):

    min_i, sums_no_padding, confidence = pattern_factory.sum_pattern.find(self.embeddings)

    self.sums = sums_no_padding
    sums = sums_no_padding

    meta = {
      'tokens': len(sums),
      'index found': min_i,
      'd-range': (sums.min(), sums.max()),
      'confidence': confidence,
      'mean': sums.mean(),
      'std': np.std(sums),
      'min': sums[min_i],
    }

    start, end = get_sentence_bounds_at_index(min_i, self.tokens)
    sentence_tokens = self.tokens[start + 1:end]

    f, sentence = extract_sum_from_tokens(sentence_tokens)

    self.found_sum = (f, (start, end), sentence, meta)

  def find_sections_indexes(self, distances_per_section_pattern, min_section_size=20):
    x = distances_per_section_pattern
    pattern_to_best_index = np.array([[idx, np.argmin(ma.masked_invalid(row))] for idx, row in enumerate(x)])

    # replace best indices with sentence starts
    pattern_to_best_index[:, 1] = self.find_sentence_beginnings(pattern_to_best_index[:, 1])

    # sort by sentence start
    pattern_to_best_index = np.sort(pattern_to_best_index.view('i8,i8'), order=['f1'], axis=0).view(np.int)

    # remove "duplicated" indexes
    return self.remove_similar_indexes(pattern_to_best_index, 1, min_section_size)

  @deprecated
  def remove_similar_indexes(self, indexes, column, min_section_size=20):
    indexes_zipped = [indexes[0]]

    for i in range(1, len(indexes)):
      if indexes[i][column] - indexes[i - 1][column] > min_section_size:
        pattern_to_token = indexes[i]
        indexes_zipped.append(pattern_to_token)
    return np.squeeze(indexes_zipped)

  def split_text_into_sections(self, paragraph_split_pattern: ExclusivePattern, min_section_size=10):

    distances_per_section_pattern, __ranges, __winning_patterns = \
      paragraph_split_pattern.calc_exclusive_distances(self.embeddings)

    # finding pattern positions
    x = distances_per_section_pattern
    indexes_zipped = self.find_sections_indexes(x, min_section_size)

    self.section_indexes = indexes_zipped

    return indexes_zipped, __ranges, __winning_patterns


class ProtocolAnlysingContext(ParsingContext):
  def __init__(self, embedder, renderer: ProtocolRenderer):
    ParsingContext.__init__(self, embedder)
    self.renderer: AbstractRenderer = renderer
    self.protocols_factory = None

    self.protocol: ProtocolDocument = None

  def process(self, text):
    self._reset_context()

    if self.protocols_factory is None:
      self.protocols_factory = ProtocolPatternFactory(self.embedder)
      self._logstep("Pattern factory created, patterns embedded into ELMO space")

    # ----
    pnames = [p.name[0:5] for p in self.protocols_factory.subject_pattern.patterns]
    c = Counter(pnames)
    subject_colormaps = ['Reds'] * c['t_dea'] + ['Greys'] * c['t_org'] + ['Greens'] * c['t_cha']
    # ----

    protocol = ProtocolDocument(text)
    print(f"ProtocolDocument text: len({len(text)})")
    protocol.parse()

    self.protocol = protocol
    protocol.embedd(self.protocols_factory)
    self._logstep("Document embedded into ELMO space")
    self.process_embedded_doc(protocol)

  def process_embedded_doc(self, doc: ProtocolDocument):

    section_indexes, __ranges, __winning_patterns = doc.split_text_into_sections(
      self.protocols_factory.paragraph_split_pattern)

    counter, ranges, winning_patterns = subject_weight_per_section(doc, self.protocols_factory.subject_pattern,
                                                                   self.protocols_factory.paragraph_split_pattern)

    section_names = [self.protocols_factory.paragraph_split_pattern.patterns[s[0]].name for s in doc.section_indexes]
    sidx = section_names.index('p_solution')
    if sidx < 0:
      sidx = section_names.index('p_agenda')
    if sidx < 0:
      sidx = section_names.index('p_question')

    if sidx < 0:
      sidx = 0

    #   html += winning_patterns_to_html(
    #       doc.tokens, ranges,
    #       winning_patterns,
    #       range(section_indexes[sidx][1], section_indexes[sidx+1][1]),
    #       colormaps=subject_colormaps )

    doc._find_sum(self.protocols_factory)
    self._logstep("value found")

    doc.per_subject_distances = None  # Hack

    self.renderer.print_results(doc)
    self.renderer.render_subject(counter)

  #   print(protocol.normal_text)

  """### Upload file code"""


"""### Define Patterns"""


class ProtocolPatternFactory(AbstractPatternFactory):
  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self._build_paragraph_split_pattern()
    self._build_subject_pattern()
    self._build_sum_margin_extraction_patterns()
    self.embedd()

  def _build_sum_margin_extraction_patterns(self):
    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = ''

    sum_comp_pat = CoumpoundFuzzyPattern()

    sum_comp_pat.add_pattern(self.create_pattern('sum_max1', (prefix + 'стоимость', 'не более 0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max2', (prefix + 'цена', 'не больше 0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max3', (prefix + 'стоимость <', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max4', (prefix + 'цена менее', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max5', (prefix + 'стоимость не может превышать', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max6', (prefix + 'общая сумма может составить', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max7', (prefix + 'лимит соглашения', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max8', (prefix + 'верхний лимит стоимости', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max9', (prefix + 'максимальная сумма', '0', suffix)))

    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg1', ('ежемесячно не позднее', '0', 'числа каждого месяца')), -0.8)
    sum_comp_pat.add_pattern(self.create_pattern('sum_max_neg2', ('приняли участие в голосовании', '0', 'человек')),
                             -0.8)
    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg3', ('срок действия не должен превышать', '0', 'месяцев с даты выдачи')), -0.8)
    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg4', ('позднее чем за', '0', 'календарных дней до даты его окончания ')), -0.8)
    sum_comp_pat.add_pattern(self.create_pattern('sum_max_neg5', ('общая площадь', '0', 'кв . м.')), -0.8)

    self.sum_pattern = sum_comp_pat

  def _build_subject_pattern(self):
    ep = ExclusivePattern()

    PRFX = "Повестка дня заседания: \n"

    if True:
      ep.add_pattern(self.create_pattern('t_deal_1', (PRFX, 'Об одобрении сделки', 'связанной с продажей')))
      ep.add_pattern(self.create_pattern('t_deal_2', (
        PRFX + 'О согласии на', 'совершение сделки', 'связанной с заключением договора')))
      ep.add_pattern(self.create_pattern('t_deal_2', (
        PRFX + 'об одобрении', 'крупной сделки', 'связанной с продажей недвижимого имущества')))

      for p in ep.patterns:
        p.soft_sliding_window_borders = True

    if True:
      ep.add_pattern(self.create_pattern('t_org1', (PRFX, 'О создании филиала', 'Общества')))
      ep.add_pattern(self.create_pattern('t_org2', (PRFX, 'Об утверждении Положения', 'о филиале Общества')))
      ep.add_pattern(self.create_pattern('t_org3', (PRFX, 'О назначении руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org4', (PRFX, 'О прекращении полномочий руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org5', (PRFX, 'О внесении изменений', '')))

    if True:
      ep.add_pattern(
        self.create_pattern('t_charity_1', (PRFX + 'О предоставлении', 'безвозмездной', 'финансовой помощи')))
      ep.add_pattern(
        self.create_pattern('t_charity_2', (PRFX + 'О согласии на совершение сделки', 'пожертвования', '')))
      ep.add_pattern(self.create_pattern('t_charity_3', (PRFX + 'Об одобрении сделки', 'пожертвования', '')))

      t_char_mix = CoumpoundFuzzyPattern()
      t_char_mix.name = "t_charity_mixed"

      t_char_mix.add_pattern(
        self.create_pattern('tm_charity_1', (PRFX + 'О предоставлении', 'безвозмездной финансовой помощи', '')))
      t_char_mix.add_pattern(
        self.create_pattern('tm_charity_2', (PRFX + 'О согласии на совершение', 'сделки пожертвования', '')))
      t_char_mix.add_pattern(self.create_pattern('tm_charity_3', (PRFX + 'Об одобрении сделки', 'пожертвования', '')))

      ep.add_pattern(t_char_mix)

    self.subject_pattern = ep

  def _build_paragraph_split_pattern(self):
    PRFX = ". \n"
    PRFX1 = """
    кворум для проведения заседания и принятия решений имеется.

    """

    sect_pt = ExclusivePattern()

    if True:
      # IDX 0
      p_agenda = CoumpoundFuzzyPattern()
      p_agenda.name = "p_agenda"

      p_agenda.add_pattern(self.create_pattern('p_agenda_1', (PRFX1, 'Повестка', 'дня заседания:')))
      p_agenda.add_pattern(self.create_pattern('p_agenda_2', (PRFX1, 'Повестка', 'дня:')))

      sect_pt.add_pattern(p_agenda)

    if True:
      # IDX 1
      p_solution = CoumpoundFuzzyPattern()
      p_solution.name = "p_solution"
      p_solution.add_pattern(
        self.create_pattern('p_solution1', (PRFX, 'решение', 'принятое по вопросу повестки дня: одобрить')))
      p_solution.add_pattern(self.create_pattern('p_solution2', (PRFX + 'формулировка', 'решения', ':одобрить')))

      sect_pt.add_pattern(p_solution)

    sect_pt.add_pattern(self.create_pattern('p_head', (PRFX, 'Протокол \n ', 'заседания')))
    sect_pt.add_pattern(self.create_pattern('p_question', (
      PRFX + 'Первый', 'вопрос', 'повестки дня заседания поставленный на голосование')))
    sect_pt.add_pattern(self.create_pattern('p_votes', (PRFX, 'Результаты голосования', 'за против воздержаолось')))
    sect_pt.add_pattern(self.create_pattern('p_addons', (PRFX, 'Приложения', '')))

    self.paragraph_split_pattern = sect_pt


"""### ProtocolDocument class"""

import numpy.ma as ma

"""## Processing"""

from collections import Counter


def subject_weight_per_section(doc, subj_pattern, paragraph_split_pattern):
  assert doc.section_indexes is not None

  distances_per_subj_pattern_, ranges_, winning_patterns = subj_pattern.calc_exclusive_distances(
    doc.embeddings)

  ranges_global = [
    np.nanmin(distances_per_subj_pattern_),
    np.nanmax(distances_per_subj_pattern_)]

  section_names = [[paragraph_split_pattern.patterns[s[0]].name, s[1]] for s in doc.section_indexes]
  voting = []
  for i in range(1, len(section_names)):
    p1 = section_names[i - 1]
    p2 = section_names[i]

    distances_per_pattern_t = distances_per_subj_pattern_[:, p1[1]:p2[1]]

    dist_per_pat = []
    for row in distances_per_pattern_t:
      dist_per_pat.append(np.nanmin(row))

    patindex = np.nanargmin(dist_per_pat)
    pat_prefix = subj_pattern.patterns[patindex].name[:5]
    #         print(patindex, pat_prefix)

    voting.append(pat_prefix)

    ## HACK more attention to particular sections
    if p1[0] == 'p_agenda' or p1[0] == 'p_solution' or p1[0] == 'p_question':
      voting.append(pat_prefix)

  return Counter(voting), ranges_global, winning_patterns
