from legal_docs import LegalDocument
from legal_docs import extract_all_contraints_from_sentence
from legal_docs import rectifyed_sum_by_pattern_prefix, tokenize_text
from ml_tools import *
from patterns import AbstractPatternFactoryLowCase
from renderer import AbstractRenderer
from transaction_values import ValueConstraint


class ContractAnlysingContext:
  def __init__(self, embedder, renderer: AbstractRenderer):
    assert embedder is not None
    assert renderer is not None

    self.verbosity_level = 2

    self.price_factory = ContractValuePatternFactory(embedder)
    self.hadlines_factory = ContractHeadlinesPatternFactory(embedder)
    self.subj_factory = ContractSubjPatternFactory(embedder)
    self.renderer = renderer

    self.contract = None
    self.contract_values = None

    self.__step = 0

  def analyze_contract(self, contract_text):
    self.__step = 0
    """
    MAIN METHOD
    
    :param contract_text: 
    :return: 
    """
    doc = ContractDocument2(contract_text)
    doc.parse()
    self.contract = doc
    self._logstep("parsing document and detecting document high-level structure")

    embedded_headlines = doc.embedd_headlines(self.hadlines_factory)
    hl_meta_by_index = doc.match_headline_types(self.hadlines_factory.headlines, embedded_headlines, 'headline.', 0.9)
    doc.sections = doc.find_sections_by_headlines(hl_meta_by_index)

    self._logstep("embedding headlines into semantic space")

    values = self.fetch_value_from_contract(doc)
    self._logstep("fetching transaction values")

    self.renderer.render_values(values)
    self.contract_values = values
    return doc, values

  def _logstep(self, name):
    s = self.__step
    print(f'❤️ ACCOMPLISHED: \t {s}.\t {name}')
    self.__step += 1

  def fetch_value_from_contract(self, contract: LegalDocument):
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
      value_section_info = sections['price.']
      value_section = value_section_info.body
      section_name = value_section_info.subdoc.untokenize_cc()
      result = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))
      if len(result) == 0:
        print(f'-WARNING: В разделе "{ section_name }" стоимость сделки не найдена!')
      if self.verbosity_level > 1:
        renderer.render_value_section_details(value_section_info)
        self._logstep(f'searching for transaction values in section  "{ section_name }"')
    else:
      print('-WARNING: Раздел про стоимость сделки не найдена!')

    if len(result) == 0:
      if 'subj' in sections:

        # fallback
        value_section_info = sections['subj']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.untokenize_cc()
        print(f'-WARNING: Ищем стоимость в разделе { section_name }')
        result = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))
        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{ section_name }"')

    if len(result) == 0:
      if 'pricecond' in sections:

        # fallback
        value_section_info = sections['pricecond']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.untokenize_cc()
        print(f'-WARNING: Ищем стоимость в разделе { section_name }!')
        result = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))
        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{ section_name }"')

    if len(result) == 0:
      print('-WARNING: Ищем стоимость во всем документе!')

      #     trying to find sum in the entire doc
      value_section = contract
      result = filter_nans(_try_to_fetch_value_from_section(value_section, price_factory))
      if self.verbosity_level > 1:
        print('ENTIRE DOC', '--' * 70)
        self._logstep(f'searching for transaction values in the entire document')
    #       render_value_section_details(value_section)

    return result


class ContractHeadlinesPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)

    self._build_head_patterns()
    self.embedd()

    self.headlines = ['subj', 'contract', 'def', 'price.', 'pricecond', 'terms', 'dates', 'break', 'rights', 'obl',
                      'resp', 'forcemajor', 'confidence', 'special', 'appl', 'addresses', 'conficts']

  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    PRFX = ''

    cp('headline.contract', (PRFX, 'ДОГОВОР',
                             '\n город, месяц, год \n общество с ограниченной ответственностью, в лице, действующего на основании, именуемое далее, заключили настоящий договор о нижеследующем'))
    cp('headline.def', (PRFX, 'Термины и определения', 'толкования'))

    cp('headline.subj.1', ('заключили настоящий Договор нижеследующем:\n' + PRFX, 'Предмет договора.\n',
                           'Исполнитель обязуется, заказчик поручает'))
    cp('headline.subj.2', (PRFX, 'ПРЕДМЕТ ДОГОВОРА', ''))

    cp('headline.price.1', (PRFX, 'цена договора', ''))
    cp('headline.price.2', (PRFX, 'СТОИМОСТЬ РАБОТ', ''))
    cp('headline.price.3', (PRFX, ' Расчеты по договору', ''))

    cp('headline.pricecond.1', (PRFX, 'УСЛОВИЯ ПЛАТЕЖЕЙ', ''))
    cp('headline.pricecond.2', (PRFX, 'Оплата услуг', ''))
    cp('headline.pricecond.3', (PRFX, 'Условия и порядок расчетов.', ''))
    cp('headline.pricecond.4', (PRFX, 'СТОИМОСТЬ УСЛУГ', ', ПОРЯДОК ИХ ПРИЕМКИ И РАСЧЕТОВ'))

    cp('headline.terms', (PRFX, 'СРОКИ ВЫПОЛНЕНИЯ РАБОТ.', 'Порядок выполнения работ.'))

    cp('headline.dates', (PRFX, 'СРОК ДЕЙСТВИЯ ДОГОВОРА.\n',
                          'настоящий договор вступает в силу с момента подписания сторонами, изменения и дополнения к договору оформляются письменным соглашением сторон, продленным на каждый последующий год'))
    cp('headline.break', (PRFX, 'Расторжение договора',
                          'досрочное расторжение договора, предупреждением о прекращении, расторгается в случаях, предусмотренных действующим законодательством, в одностороннем порядке'))

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
    cp('headline.addresses', (PRFX, 'РЕКВИЗИТЫ СТОРОН', 'ЮРИДИЧЕСКИЕ АДРЕСА'))

    cp('headline.conficts', (PRFX, 'Споры и разногласия.', ''))


class ContractValuePatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)

    self._build_sum_patterns()
    self.embedd()

  def _build_sum_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'решений о совершении сделок '

    cp('_sum.work.1', ('Стоимость Работ составляет', '0 рублей', suffix))
    cp('_sum.work.2', ('Расчеты по договору. Стоимость оказываемых услуг составляет ', '0', suffix))
    cp('_sum.work.3', ('Стоимость расчетов по договору не может превышать', '0', suffix))
    cp('_sum.work.4', ('после выставления счета оплачивает сумму в размере', '0', suffix))

    cp('sum_neg.phone', ('телефон', '00-00-00', ''))

    cp('sum_neg.penalty', ('уплачивается', 'штраф', '0 рублей а также возмещаются понесенные убытки'))
    cp('sum_neg.3', (
      'В случае нарушения  сроков выполнения Работ по соответствующему Приложению , Заказчик имеет право взыскать пени в размере',
      '0%', 'от стоимости не выполненного вовремя этапа Работ по соответствующему Приложению за каждый день просрочки'))
    cp('sum_neg.date.1', ('в срок не позднее, чем за 0 банковских', 'календарных', ' дней'))
    cp('sum_neg.vat', ('в том числе', 'НДС', '0 ' + suffix))
    cp('sum_neg.date.2', ('в течение', '0', 'рабочих дней '))

  def make_contract_value_attention_vectors(self, subdoc):
    value_attention_vector, _c1 = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, '_sum.work',
                                                                  relu_th=0.4)
    value_attention_vector = cut_above(value_attention_vector, 1)
    value_attention_vector = relu(value_attention_vector, 0.6)
    value_attention_vector = momentum(value_attention_vector, 0.8)

    novalue_attention_vector, _c1 = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, 'sum_neg',
                                                                    relu_th=0.4)
    novalue_attention_vector = cut_above(novalue_attention_vector, 1)

    novalue_attention_vector_local_contrast = relu(novalue_attention_vector, 0.6)
    novalue_attention_vector_local_contrast = momentum(novalue_attention_vector_local_contrast, 0.9)

    value_attention_vector_tuned = (value_attention_vector - novalue_attention_vector * 0.7)

    value_attention_vector_tuned = relu(value_attention_vector_tuned, 0.3)
    value_attention_vector_tuned = normalize(value_attention_vector_tuned)

    return {
      'value_attention_vector': value_attention_vector,
      'novalue_attention_vector': novalue_attention_vector,

      'novalue_attention_vector_local_contrast': novalue_attention_vector_local_contrast,
      'value_attention_vector_tuned': value_attention_vector_tuned
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


def _try_to_fetch_value_from_section(value_section: LegalDocument, factory: ContractValuePatternFactory) -> List:
  value_section.embedd(factory)
  value_section.calculate_distances_per_pattern(factory)

  # context._logstep(f'embedding for transaction values in section  "{ section_name }"')

  vectors = factory.make_contract_value_attention_vectors(value_section)

  value_section.distances_per_pattern_dict = {**value_section.distances_per_pattern_dict, **vectors}

  values: List[ValueConstraint] = extract_all_contraints_from_sentence(value_section,
                                                                       value_section.distances_per_pattern_dict[
                                                                         'value_attention_vector_tuned'])

  return values


# ----------------------------------

def filter_nans(vcs):
  r = []
  for vc in vcs:
    if not np.isnan(vc.value):
      r.append(vc)
  return r


class ContractDocument2(LegalDocument):
  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

  def tokenize(self, _txt):
    return tokenize_text(_txt)


# ------------------------------


##---------------------------------------##---------------------------------------##---------------------------------------


# self.headlines = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie', 'name']


class ContractSubjPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)
    self._build_subject_patterns()
    self.embedd()

  def _build_subject_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    cp('t_charity_1', ('договор', 'благотворительного', 'пожертвования'))
    cp('t_charity_2', ('договор  о предоставлении', 'безвозмездной  помощи', 'финансовой'))
    cp('t_charity_3', ('проведение', 'благотворительных', ''))
    cp('t_charity_4', ('', 'Благотворитель' ''))
    cp('t_charity_5', ('', 'Благополучатель' ''))
    cp('t_charity_6', ('"принимает в качестве', 'Пожертвования', ''))
    cp('t_charity_7', ('', 'Жертвователь', 'безвозмездно передает в собственность, а Благополучатель принимает'))

    cp('t_comm_1',
       ('ПРОДАВЕЦ обязуется передать в собственность ПОКУПАТЕЛЯ, а', 'ПОКУПАТЕЛЬ', 'обязуется принять и оплатить'))
    cp('t_comm_2', ('Арендодатель обязуется предоставить', 'Арендатору',
                    'за плату во временное владение и пользование недвижимое имущество '))
    cp('t_comm_3', ('Исполнитель обязуется своими силами', 'выполнить работы', 'по разработке'))
    cp('t_comm_4', ('Исполнитель обязуется', 'оказать услуги', ''))
    cp('t_comm_5', ('Заказчик поручает и оплачивает, а Исполнитель предоставляет ', 'услуги', 'в виде'))
    cp('t_comm_6', ('договор на оказание', 'платных', 'услуг'))
    cp('t_comm_7', ('договор', 'возмездного', 'оказания услуг'))
