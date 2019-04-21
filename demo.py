#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


from legal_docs import LegalDocument, HeadlineMeta
from legal_docs import extract_all_contraints_from_sentence
from legal_docs import rectifyed_sum_by_pattern_prefix, tokenize_text
from ml_tools import *
from parsing import ParsingContext, ParsingConfig
from patterns import AbstractPatternFactoryLowCase
from renderer import AbstractRenderer
from sections_finder import SectionsFinder, FocusingSectionsFinder
from transaction_values import ValueConstraint

subject_types = {
  'charity': 'благотворительность'.upper(),
  'comm': 'коммерческая сделка'.upper(),
  'comm_estate': 'недвижемость'.upper(),
  'comm_service': 'оаказание услуг'.upper()
}

subject_types_dict = {**subject_types, **{'unknown': 'предмет догоовора не ясен'}}

default_contract_parsing_config: ParsingConfig = ParsingConfig()
default_contract_parsing_config.headline_attention_threshold = 0.9


class ContractAnlysingContext(ParsingContext):
  def __init__(self, embedder, renderer: AbstractRenderer):
    ParsingContext.__init__(self, embedder, renderer)

    self.price_factory = ContractValuePatternFactory(embedder)
    self.hadlines_factory = ContractHeadlinesPatternFactory(embedder)
    self.subj_factory = ContractSubjPatternFactory(embedder)

    self.contract = None
    self.contract_values = None

    self.config = default_contract_parsing_config

    # self.sections_finder: SectionsFinder = DefaultSectionsFinder(self)
    self.sections_finder: SectionsFinder = FocusingSectionsFinder(self)

  def _reset_context(self):
    super(ContractAnlysingContext, self)._reset_context()

    if self.contract is not None:
      del self.contract
      self.contract = None

    if self.contract_values is not None:
      del self.contract_values
      self.contract_values = None

  def analyze_contract(self, contract_text):
    self._reset_context()
    """
    MAIN METHOD
    
    :param contract_text: 
    :return: 
    """
    doc = ContractDocument2(contract_text)
    doc.parse()
    self.contract = doc

    self._logstep("parsing document and detecting document high-level structure")

    self.contract.embedd(self.hadlines_factory)
    self.sections_finder.find_sections(doc, self.hadlines_factory, self.hadlines_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # -------------------------------values
    values = self.fetch_value_from_contract(doc)
    # -------------------------------subj
    doc.subject = self.recognize_subject(doc)
    self._logstep("fetching transaction values")
    # -------------------------------values

    self.renderer.render_values(values)
    self.contract_values = values
    doc.contract_values = values

    self.log_warnings()

    return doc, values

  def select_most_confident_if_almost_equal(self, a: ProbableValue, alternative: ProbableValue, m_convert,
                                            equality_range=0.0):

    if abs(m_convert(a.value).value - m_convert(alternative.value).value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
    return a

  def find_contract_best_value(self, m_convert):
    best_value: ProbableValue = max(self.contract_values,
                                    key=lambda item: m_convert(item.value).value)

    most_confident_value = max(self.contract_values, key=lambda item: item.confidence)
    best_value = self.select_most_confident_if_almost_equal(best_value, most_confident_value, m_convert,
                                                            equality_range=20)

    return best_value

  def make_subj_attention_vectors(self, subdoc, subj_types_prefixes):
    r = {}
    for subj_types_prefix in subj_types_prefixes:
      attention_vector = max_exclusive_pattern_by_prefix(subdoc.distances_per_pattern_dict, subj_types_prefix)
      attention_vector_l = relu(attention_vector, 0.6)

      r[subj_types_prefix + 'attention_vector'] = attention_vector
      r[subj_types_prefix + 'attention_vector_l'] = attention_vector_l

    return r

  def recognize_subject(self, doc):

    if 'subj' in doc.sections:
      subj_section = doc.sections['subj']

      subj_ = subj_section.body

      # ===================
      # subj_.embedd(self.subj_factory)
      subj_.calculate_distances_per_pattern(self.subj_factory)
      subj_.reset_embeddings()
      prefixes = [f't_{st}_' for st in subject_types]
      r = self.make_subj_attention_vectors(subj_, prefixes)

      interresting_vectors = [r[f't_{st}_attention_vector_l'] for st in subject_types]

      interresting_vectors_means = [np.nanmean(x) for x in interresting_vectors]
      interresting_vectors_maxes = [np.nanmax(x) for x in interresting_vectors]

      winner_id = int(np.argmax(interresting_vectors_means))

      winner_t = prefixes[winner_id][2:-1]

      confidence = interresting_vectors_maxes[winner_id]
      if confidence < 0.3:
        winner_t = 'unknown'

      return winner_t, confidence


    else:
      print('⚠️ раздел о предмете договора не найден')
      return ('unknown', 0)

  def fetch_value_from_contract(self, contract: LegalDocument) -> List[ProbableValue]:

    def filter_nans(vcs: List[ProbableValue]) -> List[ProbableValue]:
      r: List[ProbableValue] = []
      for vc in vcs:
        if vc.value is not None and not np.isnan(vc.value.value):
          r.append(vc)
      return r

    renderer = self.renderer

    price_factory = self.price_factory

    # if self.verbosity_level > 1:
    #   print('-' * 100)
    #   for eh in embedded_headlines:
    #     print(eh.untokenize_cc())

    # if self.verbosity_level > 1:
    #   print('-' * 100)
    #   for bi in hl_meta_by_index:
    #     hl = hl_meta_by_index[bi]
    #     t: LegalDocument = hl.subdoc
    #     print(bi)
    #     print('#{} \t {} \t {:.4f} \t {}'.format(hl.index, hl.type + ('.' * (14 - len(hl.type))),
    #                                              hl.confidence,
    #                                              t.untokenize_cc()
    #                                              ))
    #     renderer.render_color_text(t.tokens_cc, hl.attention_v, _range=[0, 2])

    sections = contract.sections

    result: List[ValueConstraint] = []

    if 'price.' in sections:
      value_section_info: HeadlineMeta = sections['price.']
      value_section = value_section_info.body
      section_name = value_section_info.subdoc.untokenize_cc()
      result = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))
      if len(result) == 0:
        self.warning(f'В разделе "{ section_name }" стоимость сделки не найдена!')

      if self.verbosity_level > 1:
        renderer.render_value_section_details(value_section_info)
        self._logstep(f'searching for transaction values in section  "{ section_name }"')
        # ------------
        value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
    else:
      self.warning('Раздел про стоимость сделки не найден!')

    if len(result) == 0:
      if 'subj' in sections:

        # fallback
        value_section_info = sections['subj']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.untokenize_cc()
        print(f'- Ищем стоимость в разделе { section_name }')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))

        # decrease confidence:
        for _r in result:
          _r.confidence *= 0.7

        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{ section_name }"')

        if len(result) == 0:
          self.warning(f'В разделе "{ section_name }" стоимость сделки не найдена!')

    if len(result) == 0:
      if 'pricecond' in sections:

        # fallback
        value_section_info = sections['pricecond']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.untokenize_cc()
        print(f'-WARNING: Ищем стоимость в разделе { section_name }!')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))
        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{ section_name }"')
        # ------------
        for _r in result:
          _r.confidence *= 0.7
        value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
        if len(result) == 0:
          self.warning(f'В разделе "{ section_name }" стоимость сделки не найдена!')

    if len(result) == 0:
      self.warning('Ищем стоимость во всем документе!')

      #     trying to find sum in the entire doc
      value_section = contract
      result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))
      if self.verbosity_level > 1:
        print('ENTIRE DOC', '--' * 70)
        self._logstep(f'searching for transaction values in the entire document')
      # ------------
      # decrease confidence:
      for _r in result:
        _r.confidence *= 0.6
      value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here

    return result


class ContractHeadlinesPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    # self.headlines = ['subj', 'contract', 'def', 'price.', 'pricecond', 'terms', 'dates', 'break', 'rights', 'obl',
    #                   'resp', 'forcemajor', 'confidence', 'special', 'appl', 'addresses', 'conficts']

    self.headlines = ['subj', 'contract', 'price.', 'pricecond', 'dates',
                      'resp', 'forcemajor', 'confidence', 'appl', 'addresses', 'conficts']

    AbstractPatternFactoryLowCase.__init__(self, embedder)

    self._build_head_patterns()
    self.embedd()

  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    PRFX = ''

    cp('headline.contract', (PRFX, 'ДОГОВОР',
                             '\n город, месяц, год \n общество с ограниченной ответственностью, в лице, действующего на основании, именуемое далее, заключили настоящий договор о нижеследующем'))
    cp('headline.def', (PRFX, 'Термины и определения', 'толкования'))

    cp('headline.subj.1', ('договора заключили настоящий Договор нижеследующем:', 'Предмет ',
                           'договора:\n Исполнитель обязуется, заказчик поручает'))
    cp('headline.subj.2', (PRFX, 'ПРЕДМЕТ', 'ДОГОВОРА'))
    cp('headline.subj.3', ('заключили настоящий договор о нижеследующем', 'Общие положения', ''))

    cp('headline.price.1', (PRFX, 'цена', 'договора'))
    cp('headline.price.2', (PRFX, 'СТОИМОСТЬ', 'РАБОТ'))
    cp('headline.price.3', (PRFX, ' Расчеты', 'по договору'))
    cp('headline.price.4', (PRFX, 'Оплата', 'услуг'))
    cp('headline.price.5',
       ('порядок и сроки', 'оплаты', 'согласовываются Сторонами в Дополнительных соглашениях к настоящему'))

    cp('headline.pricecond.1', ('УСЛОВИЯ ', 'ПЛАТЕЖЕЙ', ''))
    cp('headline.pricecond.3', ('Условия и порядок', 'расчетов.', ''))
    cp('headline.pricecond.4', (PRFX, 'СТОИМОСТЬ', 'УСЛУГ, ПОРЯДОК ИХ ПРИЕМКИ И РАСЧЕТОВ'))
    cp('headline.pricecond.5', (' АРЕНДНАЯ', 'ПЛАТА', 'ПОРЯДОК ВНЕСЕНИЯ АРЕНДНОЙ ПЛАТЫ'))

    cp('headline.dates.1', (PRFX, 'СРОКИ.', 'ВЫПОЛНЕНИЯ РАБОТ.Порядок выполнения работ.'))

    cp('headline.dates.2', (PRFX, 'СРОК',
                          'ДЕЙСТВИЯ. \n настоящий договор вступает в силу с момента подписания сторонами, изменения и дополнения к договору оформляются письменным соглашением сторон, продленным на каждый последующий год'))
    cp('headline.break', (PRFX, 'Расторжение',
                          'договора. \n досрочное расторжение договора, предупреждением о прекращении, расторгается в случаях, предусмотренных действующим законодательством, в одностороннем порядке'))

    cp('headline.rights.1', (PRFX, 'права и обязанности', 'сторон.\n'))
    cp('headline.obl.1', (PRFX, 'ОБЯЗАТЕЛЬСТВА', 'сторон.\n'))
    cp('headline.obl.2', (PRFX, 'ГАРАНТИЙНЫЕ', 'ОБЯЗАТЕЛЬСТВА.'))

    cp('headline.resp', (PRFX, 'Ответственность сторон.\n',
                         'невыполнения или ненадлежащего выполнения своих обязательств, несут ответственность в соответствии с действующим законодательством'))

    cp('headline.forcemajor.1', (PRFX, 'НЕПРЕОДОЛИМАЯ СИЛА.', 'ФОРС-МАЖОРНЫЕ ОБСТОЯТЕЛЬСТВА'))
    cp('headline.forcemajor.2', (PRFX, 'ОБСТОЯТЕЛЬСТВА НЕПРЕОДОЛИМОЙ СИЛЫ', ''))

    cp('headline.confidence', (PRFX, 'КОНФИДЕНЦИАЛЬНОСТЬ ИНФОРМАЦИИ.', ''))

    cp('headline.special.1', (PRFX + 'ОСОБЫЕ, дополнительные', ' УСЛОВИЯ.', ''))
    cp('headline.special.2', (PRFX, 'ЗАКЛЮЧИТЕЛЬНЫЕ ПОЛОЖЕНИЯ.', ''))

    cp('headline.appl', (PRFX, 'ПРИЛОЖЕНИЯ', 'К ДОГОВОРУ'))
    cp('headline.addresses.1', (PRFX, 'РЕКВИЗИТЫ СТОРОН', 'ЮРИДИЧЕСКИЕ АДРЕСА'))
    cp('headline.addresses.2', (PRFX, 'ЮРИДИЧЕСКИЕ АДРЕСА', 'РЕКВИЗИТЫ СТОРОН'))

    cp('headline.conficts', (PRFX, 'Споры и разногласия.', ''))


class ContractValuePatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)

    self._build_sum_patterns()
    self.embedd()

  def _build_sum_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    suffix = '(млн. тыс. миллионов тысяч рублей долларов копеек евро)'

    cp('_phrase.1', ('общая', 'сумма', 'договора составляет'))

    cp('_sum.work.1', ('Стоимость Работ составляет', '0 рублей', suffix))
    cp('_sum.work.2', ('Расчеты по договору. Стоимость оказываемых услуг составляет ', '0', suffix))
    cp('_sum.work.3', ('Стоимость расчетов по договору не может превышать', '0', suffix))
    cp('_sum.work.4', ('после выставления счета оплачивает сумму в размере', '0', suffix))
    cp('_sum.work.5', ('Общая сумма договора составляет', '0', suffix))

    cp('sum_neg.phone', ('телефон', '00-00-00', ''))

    cp('sum_neg.penalty', ('уплачивается', 'штраф', '0 рублей а также возмещаются понесенные убытки'))
    cp('sum_neg.3', (
      'В случае нарушения  сроков выполнения Работ по соответствующему Приложению , Заказчик имеет право взыскать пени в размере',
      '0%', 'от стоимости не выполненного вовремя этапа Работ по соответствующему Приложению за каждый день просрочки'))
    cp('sum_neg.date.1', ('в срок не позднее, чем за 0 банковских', 'календарных', ' дней'))
    cp('sum_neg.vat', ('в том числе', 'НДС', '0 ' + suffix))
    cp('sum_neg.date.2', ('в течение', '0', 'рабочих дней '))

  def make_contract_value_attention_vectors(self, subdoc):
    sumphrase_attention_vector = max_exclusive_pattern_by_prefix(subdoc.distances_per_pattern_dict, '_phrase')
    sumphrase_attention_vector = momentum(sumphrase_attention_vector, 0.99)

    value_attention_vector, _c1 = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, '_sum.work',
                                                                  relu_th=0.4)
    value_attention_vector = cut_above(value_attention_vector, 1)
    value_attention_vector = relu(value_attention_vector, 0.6)
    value_attention_vector = momentum(value_attention_vector, 0.8)

    novalue_attention_vector = max_exclusive_pattern_by_prefix(subdoc.distances_per_pattern_dict, 'sum_neg')

    novalue_attention_vector_local_contrast = relu(novalue_attention_vector, 0.6)
    novalue_attention_vector_local_contrast = momentum(novalue_attention_vector_local_contrast, 0.9)

    value_attention_vector_tuned = (value_attention_vector - novalue_attention_vector * 0.7)

    value_attention_vector_tuned = (value_attention_vector_tuned + sumphrase_attention_vector) / 2
    value_attention_vector_tuned = relu(value_attention_vector_tuned, 0.2)

    return {
      'sumphrase_attention_vector': sumphrase_attention_vector,
      'value_attention_vector': value_attention_vector,
      'novalue_attention_vector': novalue_attention_vector,

      'novalue_attention_vector_local_contrast': novalue_attention_vector_local_contrast,
      'value_attention_vector_tuned': value_attention_vector_tuned,

    }


# ----------------------------------------------------------------------------------------------
def subdoc_between_lines(line_a: int, line_b: int, doc):
  _str = doc.structure.structure
  start = _str[line_a].span[1]
  if line_b is not None:
    end = _str[line_b].span[0]
  else:
    end = len(doc.tokens)

  return doc.subdoc(start, end)


# ----------------------------------------------------------------------------------------------


def _try_to_fetch_value_from_section(value_section: LegalDocument, factory: ContractValuePatternFactory) -> List[
  ProbableValue]:
  # value_section.embedd(factory)
  value_section.calculate_distances_per_pattern(factory)

  # context._logstep(f'embedding for transaction values in section  "{ section_name }"')

  vectors = factory.make_contract_value_attention_vectors(value_section)

  value_section.distances_per_pattern_dict = {**value_section.distances_per_pattern_dict, **vectors}

  values: List[ProbableValue] = extract_all_contraints_from_sentence(value_section,
                                                                     value_section.distances_per_pattern_dict[
                                                                       'value_attention_vector_tuned'])

  return values


# ----------------------------------


class ContractDocument2(LegalDocument):
  def __init__(self, original_text: str):
    LegalDocument.__init__(self, original_text)
    self.subject = ('unknown', 1.0)
    self.contract_values = [ProbableValue]

  def tokenize(self, _txt):
    return tokenize_text(_txt)


##---------------------------------------##---------------------------------------##---------------------------------------


class ContractSubjPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)
    self._build_subject_patterns()
    self.embedd()

  def _build_subject_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    cp('t_charity_1', ('договор',
                       'благотворительного',
                       'пожертвования'))

    cp('t_charity_2', ('договор о предоставлении',
                       'безвозмездной помощи',
                       'финансовой'))

    cp('t_charity_3', ('проведение',
                       'благотворительных',
                       ''))

    cp('t_charity_4', ('', 'Благотворитель', ''))
    cp('t_charity_5', ('', 'Благополучатель', ''))

    cp('t_charity_6', ('принимает в качестве',
                       'Пожертвования',
                       ''))

    cp('t_charity_7', ('',
                       'Жертвователь',
                       'безвозмездно передает в собственность, а Благополучатель принимает'))

    cp('t_charity_8', ('Жертвователь', 'безвозмездно', ''))

    cp('t_comm_1',
       ('ПРОДАВЕЦ обязуется передать в собственность ПОКУПАТЕЛЯ, а', 'ПОКУПАТЕЛЬ', 'обязуется принять и оплатить'))
    cp('t_comm_estate_2', ('Арендодатель обязуется предоставить',
                           'Арендатору',
                           'за плату во временное владение и пользование недвижимое имущество '))

    cp('t_comm_service_3', ('Исполнитель обязуется своими силами',
                            'выполнить работы',
                            'по разработке'))

    cp('t_comm_service_4', ('Исполнитель обязуется',
                            'оказать услуги',
                            ''))

    cp('t_comm_service_5', ('Заказчик поручает и оплачивает, а Исполнитель предоставляет ', 'услуги', 'в виде'))
    cp('t_comm_service_6', ('договор на оказание', 'платных', 'услуг'))
    cp('t_comm_service_7', ('договор', 'возмездного', 'оказания услуг'))
