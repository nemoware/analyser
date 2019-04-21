import numpy as np

from charter_patterns import CharterPatternFactory
from legal_docs import CharterDocument, HeadlineMeta, LegalDocument, \
  make_constraints_attention_vectors, extract_all_contraints_from_sentence, \
  deprecated, org_types, embedd_generic_tokenized_sentences_2
from ml_tools import split_by_token
from parsing import ParsingContext, ParsingConfig, head_types_dict
from renderer import *
from text_tools import find_ner_end
from text_tools import untokenize
from transaction_values import extract_sum, ValueConstraint

default_charter_parsing_config:ParsingConfig=ParsingConfig()
default_charter_parsing_config.headline_attention_threshold = 1.4


class CharterAnlysingContext(ParsingContext):
  def __init__(self, embedder, renderer: AbstractRenderer):
    ParsingContext.__init__(self, embedder, renderer)

    self.factory: CharterPatternFactory = None

    self.renderer = renderer

    self.org = None
    self.constraints = None
    self.doc = None

    self.config = default_charter_parsing_config

  def analyze_charter(self, txt, verbose=False):
    """
    🚀
    :param txt:
    :param verbose:
    :return:
    """

    if self.factory is None:
      self.factory = CharterPatternFactory(self.embedder)

    self._reset_context()
    # 0. parse
    _charter_doc = CharterDocument(txt)
    _charter_doc.right_padding = 0

    # 1. find top level structure
    _charter_doc.parse()

    self.doc = _charter_doc

    # 2. embedd headlines
    embedded_headlines = _charter_doc.embedd_headlines(self.factory)
    self._logstep("embedding headlines into semantic space")

    _charter_doc.sections = _charter_doc.find_sections_by_headlines_2(
      self, self.factory.headlines, embedded_headlines, 'headline.', self.config.headline_attention_threshold)

    self._logstep("extracting doc structure")

    if 'name' in _charter_doc.sections:
      section: HeadlineMeta = _charter_doc.sections['name']
      org = self.detect_ners(section.body)
      self._logstep("extracting NERs (named entities)")
    else:
      self.warning('Секция наименования компнании не найдена')
      self.warning('Попытаемся искать просто в начале документа')
      org = self.detect_ners(_charter_doc.subdoc(0, 3000))
      # org = {
      #   'type': 'org_unknown',
      #   'name': "не определено",
      #   'type_name': "не определено",
      #   'tokens': [],
      #   'attention_vector': []
      # }

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
  @deprecated
  def detect_ners(self, section):
    """
    XXX: TODO: 🚷🔥 moved to charter_parser.py
    :param section:
    :return:
    """
    assert section is not None

    section.embedd(self.factory)

    section.calculate_distances_per_pattern(self.factory, pattern_prefix='org_', merge=True)
    section.calculate_distances_per_pattern(self.factory, pattern_prefix='ner_org', merge=True)
    section.calculate_distances_per_pattern(self.factory, pattern_prefix='nerneg_', merge=True)

    org_by_type_dict, org_type = self._detect_org_type_and_name(section)

    if self.verbosity_level > 1:
      self.renderer.render_color_text(section.tokens_cc, section.distances_per_pattern_dict[org_type],
                                      _range=[0, 1])

    start = org_by_type_dict[org_type][0]
    start = start + len(self.factory.patterns_dict[org_type].embeddings)
    end = 1 + find_ner_end(section.tokens, start)

    orgname_sub_section: LegalDocument = section.subdoc(start, end)
    org_name = orgname_sub_section.untokenize_cc()

    if self.verbosity_level > 1:
      self.renderer.render_color_text(orgname_sub_section.tokens_cc,
                                      orgname_sub_section.distances_per_pattern_dict[org_type],
                                      _range=[0, 1])
      print('Org type:', org_types[org_type], org_by_type_dict[org_type])

    rez = {
      'type': org_type,
      'name': org_name,
      'type_name': org_types[org_type],
      'tokens': section.tokens_cc,
      'attention_vector': section.distances_per_pattern_dict[org_type]
    }

    return rez

  # ------------------------------------------------------------------------------
  @deprecated
  def _detect_org_type_and_name(self, section):
    """
        XXX: TODO: 🚷🔥 moved to charter_parser.py

    """
    s_attention_vector_neg = self.factory._build_org_type_attention_vector(section)

    org_by_type = {}
    best_org_type = None
    _max = 0
    for org_type in org_types.keys():

      vector = section.distances_per_pattern_dict[org_type] * s_attention_vector_neg
      if self.verbosity_level > 2:
        print('_detect_org_type_and_name, org_type=', org_type, section.distances_per_pattern_dict[org_type][0:10])

      idx = np.argmax(vector)
      val = section.distances_per_pattern_dict[org_type][idx]
      if val > _max:
        _max = val
        best_org_type = org_type

      org_by_type[org_type] = [idx, val]

    if self.verbosity_level > 2:
      print('_detect_org_type_and_name', org_by_type)

    return org_by_type, best_org_type

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
      'sentences': self._extract_constraint_values_from_region(sentenses_i, self.factory)
    }
    self._logstep(f"Finding margin transaction values in section {untokenize(hl_subdoc.tokens_cc)}")
    return r_by_head_type

  ##---------------------------------------
  def _extract_constraint_values_from_region(self, sentenses_i, _embedd_factory):
    if sentenses_i is None or len(sentenses_i) == 0:
      return []

    ssubdocs = embedd_generic_tokenized_sentences_2(sentenses_i, _embedd_factory.embedder)

    for ssubdoc in ssubdocs:
      ssubdoc.calculate_distances_per_pattern(_embedd_factory, pattern_prefix='sum_max.', merge=True)
      ssubdoc.calculate_distances_per_pattern(_embedd_factory, pattern_prefix='sum__', merge=True)
      ssubdoc.calculate_distances_per_pattern(_embedd_factory, pattern_prefix='d_order.', merge=True)

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
      return as_error_html("сумма контракта неизвестна")
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
