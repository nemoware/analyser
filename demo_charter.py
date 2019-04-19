import numpy as np

from legal_docs import CharterDocument, HeadlineMeta, LegalDocument, \
  embedd_generic_tokenized_sentences, make_constraints_attention_vectors, extract_all_contraints_from_sentence, \
  deprecated, make_soft_attention_vector, org_types, ParsingContext
from ml_tools import split_by_token
from patterns import AbstractPatternFactoryLowCase, AbstractPatternFactory, FuzzyPattern
from renderer import *
from text_tools import find_ner_end
from text_tools import untokenize
from transaction_values import extract_sum, ValueConstraint


class CharterAnlysingContext(ParsingContext):
  def __init__(self, embedder, renderer: AbstractRenderer):
    ParsingContext.__init__(self, embedder, renderer)

    self.price_factory = None
    self.hadlines_factory: CharterHeadlinesPatternFactory = None
    self.ner_factory: CharterNerPatternFactory = None
    self.renderer = renderer

    self.org = None
    self.constraints = None
    self.doc = None

  def analyze_charter(self, txt, verbose=False):

    if self.price_factory is None:
      self.price_factory = CharterConstraintsPatternFactory(self.embedder)

    if self.hadlines_factory is None:
      self.hadlines_factory = CharterHeadlinesPatternFactory(self.embedder)

    if self.ner_factory is None:
      self.ner_factory = CharterNerPatternFactory(self.embedder)

    self._reset_context()
    # parse
    _charter_doc = CharterDocument(txt)
    _charter_doc.right_padding = 0
    # 1. find top level structure
    _charter_doc.parse()

    self.doc = _charter_doc

    # 2. embedd headlines
    embedded_headlines = _charter_doc.embedd_headlines(self.hadlines_factory)
    self._logstep("embedding headlines into semantic space")

    _charter_doc.sections = _charter_doc.find_sections_by_headlines_2(
      self, self.hadlines_factory.headlines, embedded_headlines, 'headline.', 1.4)

    self._logstep("extracting doc structure")

    if 'name' in _charter_doc.sections:
      section: HeadlineMeta = _charter_doc.sections['name']
      org = self.detect_ners(section.body)
      self._logstep("extracting NERs (named entities)")
    else:
      self.warning('Секция наименования компнании не найдена')
      org = {
        'type': 'org_unknown',
        'name': "не определено",
        'type_name': "не определено",
        'tokens': [],
        'attention_vector': []
      }

    rz = self.find_contraints(_charter_doc.sections)
    self._logstep("Finding margin transaction values")

    #   html = render_constraint_values(rz)
    #   display(HTML(html))
    self.org = org
    self.constraints = rz

    self.verbosity_level = 1

    self.log_warnings()

    return org, rz

  # ------------------------------------------------------------------------------
  def detect_ners(self, section):
    assert section is not None

    section.embedd(self.ner_factory)
    section.calculate_distances_per_pattern(self.ner_factory)

    dict_org, best_type = self._detect_org_type_and_name(section)

    if self.verbosity_level > 1:
      self.renderer.render_color_text(section.tokens_cc, section.distances_per_pattern_dict[best_type],
                                      _range=[0, 1])

    start = dict_org[best_type][0]
    start = start + len(self.ner_factory.patterns_dict[best_type].embeddings)
    end = 1 + find_ner_end(section.tokens, start)

    orgname_sub_section: LegalDocument = section.subdoc(start, end)
    org_name = orgname_sub_section.untokenize_cc()

    if self.verbosity_level > 1:
      self.renderer.render_color_text(orgname_sub_section.tokens_cc,
                                      orgname_sub_section.distances_per_pattern_dict[best_type],
                                      _range=[0, 1])
      print('Org type:', org_types[best_type], dict_org[best_type])

    rez = {
      'type': best_type,
      'name': org_name,
      'type_name': org_types[best_type],
      'tokens': section.tokens_cc,
      'attention_vector': section.distances_per_pattern_dict[best_type]
    }

    return rez

  # ------------------------------------------------------------------------------
  def _detect_org_type_and_name(self, section):
    s_attention_vector_neg = self.ner_factory._build_org_type_attention_vector(section)

    dict_org = {}
    best_type = None
    _max = 0
    for org_type in org_types.keys():

      vector = section.distances_per_pattern_dict[org_type] * s_attention_vector_neg
      if self.verbosity_level > 2:
        print('_detect_org_type_and_name, org_type=', org_type, section.distances_per_pattern_dict[org_type][0:10])

      idx = np.argmax(vector)
      val = section.distances_per_pattern_dict[org_type][idx]
      if val > _max:
        _max = val
        best_type = org_type

      dict_org[org_type] = [idx, val]

    if self.verbosity_level > 2:
      print('_detect_org_type_and_name', dict_org)

    return dict_org, best_type

  # ---------------------------------------
  def find_contraints(self, sections):
    # 5. extract constraint values
    sections_filtered = {}
    prefix = 'head.'
    for k in sections:
      if k[:len(prefix)] == prefix:
        sections_filtered[k] = sections[k]

    rz = self.extract_constraint_values_from_sections(sections_filtered)
    return rz

  ##---------------------------------------
  def extract_constraint_values_from_sections(self, sections):
    rez = {}

    for head_type in sections:
      section = sections[head_type]
      rez[head_type] = self.extract_constraint_values_from_section(section)

    return rez

  ##---------------------------------------
  def extract_constraint_values_from_section(self, section: HeadlineMeta):

    if self.verbosity_level > 1:
      print('extract_constraint_values_from_section', section.type)

    body = section.body

    if self.verbosity_level > 1:
      print('extract_constraint_values_from_section', 'embedding....')

    sentenses_i = []
    senetences = split_by_token(body.tokens, '\n')
    for s in senetences:
      line = untokenize(s) + '\n'
      sum = extract_sum(line)
      if sum is not None:
        sentenses_i.append(line)
      if self.verbosity_level > 2:
        print('-', sum, line)

    hl_subdoc = section.subdoc

    r_by_head_type = {
      'section': head_types_dict[section.type],
      'caption': untokenize(hl_subdoc.tokens_cc),
      'sentences': self._extract_constraint_values_from_region(sentenses_i, self.price_factory)
    }
    self._logstep(f"Finding margin transaction values in section {untokenize(hl_subdoc.tokens_cc)}")
    return r_by_head_type

  ##---------------------------------------
  def _extract_constraint_values_from_region(self, sentenses_i, _embedd_factory):
    if sentenses_i is None or len(sentenses_i) == 0:
      return []

    ssubdocs = embedd_generic_tokenized_sentences(sentenses_i, _embedd_factory)

    for ssubdoc in ssubdocs:

      vectors = make_constraints_attention_vectors(ssubdoc)
      ssubdoc.distances_per_pattern_dict = {**ssubdoc.distances_per_pattern_dict, **vectors}

      if self.verbosity_level > 1:
        self.renderer.render_color_text(
          ssubdoc.tokens,
          ssubdoc.distances_per_pattern_dict['deal_value_attention_vector'], _range=(0, 1))

    sentences = []
    for sentence_subdoc in ssubdocs:
      constraints: List[ValueConstraint] = extract_all_contraints_from_sentence(sentence_subdoc,
                                                                                sentence_subdoc.distances_per_pattern_dict[
                                                                                  'deal_value_attention_vector'])

      sentence = {
        'quote': untokenize(sentence_subdoc.tokens_cc),
        'subdoc': sentence_subdoc,
        'constraints': constraints
      }

      sentences.append(sentence)
    return sentences

  # ==============
  # VIOLATIONS

  def find_ranges_by_group(self, charter_constraints, m_convert, verbose=False):
    ranges_by_group = {}
    for head_group in charter_constraints:
      #     print('-' * 20)
      group_c = charter_constraints[head_group]
      data = self._combine_constraints_in_group(group_c, m_convert, verbose)
      ranges_by_group[head_group] = data
    return ranges_by_group

  def _combine_constraints_in_group(self, group_c, m_convert, verbose=False):
    # print(group_c)
    # print(group_c['section'])

    data = {
      'name': group_c['section'],
      'ranges': {}
    }

    sentences = group_c['sentences']
    #   print (charter_constraints[head_group]['sentences'])
    sentence_id = 0
    for sentence in sentences:
      constraint_low = None
      constraint_up = None

      sentence_id += 1
      #     print (sentence['constraints'])

      s_constraints = sentence['constraints']
      # большие ищем
      maximals = [x for x in s_constraints if x.value.sign > 0]

      if len(maximals) > 0:
        constraint_low = min(maximals, key=lambda item: m_convert(item.value).value)
        if verbose:
          print("all maximals:")
          self.renderer.render_values(maximals)
          print('\t\t\t constraint_low', constraint_low.value.value)
          self.renderer.render_values([constraint_low])

      minimals = [x for x in s_constraints if x.value.sign <= 0]
      if len(minimals) > 0:
        constraint_up = min(minimals, key=lambda item: m_convert(item.value).value)
        if verbose:
          print("all: minimals")
          self.renderer.render_values(minimals)
          print('\t\t\t constraint_upper', constraint_up.value.value)
          self.renderer.render_values([constraint_up])
          print("----X")

      if constraint_low is not None or constraint_up is not None:
        data['ranges'][sentence_id] = VConstraint(constraint_low, constraint_up, group_c)

    return data
  # ==================================================================VIOLATIONS


class CharterHeadlinesPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)
    self._build_head_patterns()
    self.embedd()

    self.headlines = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie', 'name']

  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    head_prfx = "статья 0"

    cp('headline.name.1', ('Полное', 'фирменное наименование', 'общества на русском языке:'))
    cp('headline.name.2', ('', 'ОБЩИЕ ПОЛОЖЕНИЯ', ''))
    cp('headline.name.3', ('', 'фирменное', ''))
    cp('headline.name.4', ('', 'русском', ''))
    cp('headline.name.5', ('', 'языке', ''))
    cp('headline.name.6', ('', 'полное', ''))

    cp('headline.head.all.1', (head_prfx, 'компетенции общего собрания акционеров', ''))
    cp('headline.head.all.2', (head_prfx, 'компетенции общего собрания участников', 'общества'))
    cp('headline.head.all.3', (head_prfx, 'собрание акционеров\n', ''))

    cp('headline.head.all.4', ('', 'компетенции', ''))
    cp('headline.head.all.5', ('', 'собрания', ''))
    cp('headline.head.all.6', ('', 'участников', ''))
    cp('headline.head.all.7', ('', 'акционеров', ''))

    cp('headline.head.directors.1', (head_prfx, 'компетенция совета директоров', 'общества'))
    cp('headline.head.directors.2', ('', 'совет директоров общества', ''))
    cp('headline.head.directors.3', ('', 'компетенции', ''))
    cp('headline.head.directors.4', ('', 'совета', ''))
    cp('headline.head.directors.5', ('', 'директоров', ''))

    cp('headline.head.pravlenie.1', (head_prfx, 'компетенции правления', ''))
    cp('headline.head.pravlenie.2', ('', 'компетенции', ''))
    cp('headline.head.pravlenie.3', ('', 'правления', ''))
    #     cp('d_head_pravlenie.2', ('', 'общества', ''))

    cp('headline.head.gen.1', (head_prfx, 'компетенции генерального директора', ''))
    cp('headline.head.gen.2', ('', 'компетенции', ''))
    cp('headline.head.gen.3', ('', 'генерального', ''))
    cp('headline.head.gen.4', ('', 'директора', ''))


class CharterConstraintsPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)

    self._build_order_patterns()
    self._build_sum_margin_extraction_patterns()
    self._build_sum_patterns()
    self.embedd()

  def _build_order_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    prefix = 'принятие решения о согласии на совершение или о последующем одобрении'

    cp('d_order_4', (prefix, 'cделки', ', стоимость которой равна или превышает'))
    cp('d_order_5', (prefix, 'cделки', ', стоимость которой составляет менее'))

  def _build_sum_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'решений о совершении сделок '

    cp('sum_max1', (prefix + 'стоимость', 'не более 0', suffix))
    cp('sum_max2', (prefix + 'цена', 'не больше 0', suffix))
    cp('sum_max3', (prefix + 'стоимость <', '0', suffix))
    cp('sum_max4', (prefix + 'цена менее', '0', suffix))
    cp('sum_max5', (prefix + 'стоимость не может превышать', '0', suffix))
    cp('sum_max6', (prefix + 'общая сумма может составить', '0', suffix))
    cp('sum_max7', (prefix + 'лимит соглашения', '0', suffix))
    cp('sum_max8', (prefix + 'верхний лимит стоимости', '0', suffix))
    cp('sum_max9', (prefix + 'максимальная сумма', '0', suffix))

  def _build_sum_margin_extraction_patterns(self):
    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'совершение сделок '

    # less than
    self.create_pattern('sum__lt_1', (prefix + 'стоимость', 'не более 0', suffix))
    self.create_pattern('sum__lt_2', (prefix + 'цена', 'не больше 0', suffix))
    self.create_pattern('sum__lt_3', (prefix + 'стоимость', '< 0', suffix))
    self.create_pattern('sum__lt_4', (prefix + 'цена', 'менее 0', suffix))
    self.create_pattern('sum__lt_4.1', (prefix + 'цена', 'ниже 0', suffix))
    self.create_pattern('sum__lt_5', (prefix + 'стоимость', 'не может превышать 0', suffix))
    self.create_pattern('sum__lt_6', (prefix + 'лимит соглашения', '0', suffix))
    self.create_pattern('sum__lt_7', (prefix + 'верхний лимит стоимости', '0', suffix))
    self.create_pattern('sum__lt_8', (prefix, 'максимум 0', suffix))
    self.create_pattern('sum__lt_9', (prefix, 'до 0', suffix))
    self.create_pattern('sum__lt_10', (prefix, 'но не превышающую 0', suffix))
    self.create_pattern('sum__lt_11', (prefix, 'совокупное пороговое значение 0', suffix))

    # greather than
    self.create_pattern('sum__gt_1', (prefix + 'составляет', 'более 0', suffix))
    self.create_pattern('sum__gt_2', (prefix + '', 'превышает 0', suffix))
    self.create_pattern('sum__gt_3', (prefix + '', 'свыше 0', suffix))
    self.create_pattern('sum__gt_4', (prefix + '', 'сделка имеет стоимость, равную или превышающую 0', suffix))


class CharterPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)

    self._build_order_patterns()
    self.embedd()

  @deprecated
  def _build_order_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    cp('d_order_1', ('Порядок', 'одобрения сделок', 'в совершении которых имеется заинтересованность'))
    cp('d_order_2', ('', 'принятие решений', 'о совершении сделок'))
    cp('d_order_3',
       ('', 'одобрение заключения', 'изменения или расторжения какой-либо сделки Общества'))
    cp('d_order_4', ('', 'Сделки', 'стоимость которой равна или превышает'))
    cp('d_order_5', ('', 'Сделки', 'стоимость которой составляет менее'))


class CharterNerPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self.patterns_dict = {}

    self._build_ner_patterns()
    self.embedd()

  def _build_ner_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    for o_type in org_types.keys():
      cp(o_type, ('', org_types[o_type], '"'))

    cp('ner_org.1', ('Полное', 'фирменное наименование', 'общества на русском языке:'))

    cp('ner_org.6', ('', 'ОБЩИЕ ПОЛОЖЕНИЯ', ''))

    cp('ner_org.2', ('', 'фирменное', ''))
    cp('ner_org.3', ('', 'русском', ''))
    cp('ner_org.4', ('', 'языке', ''))
    cp('ner_org.5', ('', 'полное', ''))

    cp('nerneg_1', ('общество имеет', 'печать', ''))
    cp('nerneg_2', ('', 'сокращенное', ''))
    cp('nerneg_3', ('на', 'английском', 'языке'))

  def _build_org_type_attention_vector(self, subdoc: CharterDocument):
    attention_vector_neg = make_soft_attention_vector(subdoc, 'nerneg_1', blur=80)
    attention_vector_neg = 1 + (1 - attention_vector_neg)  # normalize(attention_vector_neg * -1)
    return attention_vector_neg


head_types_dict = {'head.directors': 'Совет директоров',
                   'head.all': 'Общее собрание участников/акционеров',
                   'head.gen': 'Генеральный директор',
                   #                      'shareholders':'Общее собрание акционеров',
                   'head.pravlenie': 'Правление общества',
                   'head.unknown': '*Неизвестный орган управления*'}
head_types = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie']

# =======================
# =======================
# =======================
from ml_tools import ProbableValue


class VConstraint:
  def __init__(self, lower, upper, head_group):
    self.lower = ProbableValue(ValueConstraint(0, 'RUB', +1), 0)
    self.upper = ProbableValue(ValueConstraint(np.inf, 'RUB', -1), 0)

    if lower is not None:
      self.lower = lower

    if upper is not None:
      self.upper = upper

    self.head_group = head_group

  def maybe_convert(self, v: ValueConstraint, convet_m):
    html = ""
    v_converted = v
    if v.currency != 'RUB':
      v_converted = convet_m(v)
      html += as_warning(f"конвертация валют {as_currency(v)} --> RUB ")
      html += as_offset(as_warning(f"примерно: {as_currency(v)} ~~  {as_currency(v_converted)}  "))
    return v, v_converted, html

  def check_contract_value(self, _v: ProbableValue, convet_m, renderer):
    greather_lower = False
    greather_upper = False

    if _v is None:
      return as_error_html("сумма контракта не известна")
    v: ValueConstraint = _v.value

    if v is None:
      return as_error_html("сумма контракта не верна")

    if v.value is None:
      return as_error_html(f"сумма контракта не верна {v.currency}")
    ###----

    lower_v = None
    upper_v = None
    if self.lower is not None:
      lower_v: ValueConstraint = self.lower.value
    if self.upper is not None:
      upper_v: ValueConstraint = self.upper.value

    html = as_msg(f"диапазон: {as_currency(lower_v)} < ..... < {as_currency(upper_v)}")

    v, v_converted, h = self.maybe_convert(v, convet_m)
    html += h

    if self.lower is not None:
      lower_v: ValueConstraint = self.lower.value
      lower_v, lower_converted, h = self.maybe_convert(lower_v, convet_m)
      html += h

      if v_converted.value >= lower_converted.value:
        greather_lower = True
        html += as_warning("требуется одобрение...".upper())
        html += as_warning(
          f"сумма договора  {as_currency(v_converted)}  БОЛЬШЕ нижней пороговой {as_currency(lower_converted)} ")
        html += as_quote(untokenize(lower_v.context[0]))

    if self.upper is not None:

      upper_v: ValueConstraint = self.upper.value
      upper_v, upper_converted, h = self.maybe_convert(upper_v, convet_m)
      html += h

      if v_converted.value >= upper_converted.value:

        html += as_error_html(
          f"сумма договора  {as_currency(v_converted)} БОЛЬШЕ верхней пороговой {as_currency(upper_converted)} ")

      elif greather_lower:
        head_name = self.head_group['section']
        html += as_error_html(f'требуется одобрение со стороны "{head_name.upper()}"')

        if lower_v.context is not None:
          html += as_quote(renderer.to_color_text(lower_v.context[0], lower_v.context[1], _range=[0, 1]))
        if upper_v.context is not None:
          html += '<br>'
          html += as_quote(renderer.to_color_text(upper_v.context[0], upper_v.context[1], _range=[0, 1]))

    return html

# -----------


# rendering:----------------------------
