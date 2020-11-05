from analyser.legal_docs import rectifyed_sum_by_pattern_prefix
from analyser.ml_tools import max_exclusive_pattern_by_prefix, momentum, cut_above, relu
from analyser.patterns import AbstractPatternFactoryLowCase
from analyser.structures import ContractSubject
from tf_support.embedder_elmo import ElmoEmbedder

contract_headlines_patterns = {
  'КУПЛИ-ПРОДАЖИ НЕДВИЖИМОГО ИМУЩЕСТВА': ContractSubject.RealEstate,
  'КУПЛИ-ПРОДАЖИ НЕДВИЖИМОСТИ': ContractSubject.RealEstate,

  'ПОСТАВКИ': ContractSubject.Deal,
  'ЛИЦЕНЗИОННЫЙ': ContractSubject.Deal,
  'СУБЛИЦЕНЗИОННЫЙ': ContractSubject.Deal,

  'займа': ContractSubject.Loans,

  'агентирования': ContractSubject.AgencyContract,
  'агентский': ContractSubject.AgencyContract,

  'оказания консультационных услуг': ContractSubject.Service,
  'на оказание услуг': ContractSubject.Service,
  'ВОЗМЕЗДНОГО ОКАЗАНИЯ УСЛУГ': ContractSubject.Service,
  'на выполнение работ по разработке информационных систем': ContractSubject.Service,
  'НА ВЫПОЛНЕНИЕ ИНЖЕНЕРНО-ИЗЫСКАТЕЛЬСКИХ РАБОТ': ContractSubject.Service,
  'НА ТЕХНИЧЕСКОЕ ОБСЛУЖИВАНИЕ И РЕМОНТ': ContractSubject.Service,
  'НА Разработку': ContractSubject.Service,

  'залога': ContractSubject.PledgeEncumbrance,

  'о безвозмездной помощи ( Пожертвование )': ContractSubject.Charity,
  'пожертвования': ContractSubject.Charity,
  'целевого пожертвования': ContractSubject.Charity,
  'благотворительной помощи': ContractSubject.Charity,
  'ДАРЕНИЯ': ContractSubject.Charity,
  'дарения движимого имущества': ContractSubject.Charity,
  'безвозмездного пользования нежилым помещением': ContractSubject.Charity,


  'генерального подряда': ContractSubject.GeneralContract,
  'подряда': ContractSubject.GeneralContract,

  'аренды недвижимого имущества': ContractSubject.Renting,
  'аренды': ContractSubject.Renting,

  'страхования': ContractSubject.Insurance
}
head_subject_patterns_prefix = 'hds_'


class ContractPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder=None):
    AbstractPatternFactoryLowCase.__init__(self)
    # self.headlines = ['subj', 'contract', 'def', 'price.', 'pricecond', 'terms', 'dates', 'break', 'rights', 'obl',
    #                   'resp', 'forcemajor', 'security', 'special', 'appl', 'addresses', 'conficts']

    self.headlines = ['subj', 'contract', 'cvalue', 'pricecond', 'dates',
                      'resp', 'forcemajor', 'security', 'appl', 'addresses', 'conficts', 'obl']

    self._build_head_patterns()
    self._build_head_subject_patterns()
    self._build_sum_patterns()
    self._build_subject_patterns()

    if embedder is not None:
      self.embedd(embedder)

  def _build_head_subject_patterns(self):

    for txt in contract_headlines_patterns:
      cnt = len(self.patterns)
      self.create_pattern(f'{head_subject_patterns_prefix}{contract_headlines_patterns[txt]}.{cnt}',
                          ('Договор', txt.lower(), ''))

  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    p_r_f_x = ''

    cp('headline.contract', (p_r_f_x, 'ДОГОВОР',
                             '\n город, месяц, год \n общество с ограниченной ответственностью, в лице, действующего на основании, именуемое далее, заключили настоящий договор о нижеследующем'))
    cp('headline.def', (p_r_f_x, 'Термины и определения', 'толкования'))


    cp('headline.subj.1', ('договора заключили настоящий Договор нижеследующем:', 'Предмет ',
                           'договора:\n Исполнитель обязуется, заказчик поручает'))
    cp('headline.subj.2', (p_r_f_x, 'ПРЕДМЕТ', 'ДОГОВОРА'))
    cp('headline.subj.3', ('заключили настоящий договор о нижеследующем', 'Общие положения', ''))

    cp('headline.cvalue.1', (p_r_f_x, 'цена', 'договора'))
    cp('headline.cvalue.2', (p_r_f_x, 'СТОИМОСТЬ', 'РАБОТ'))
    cp('headline.cvalue.3', (p_r_f_x, ' Расчеты', 'по договору'))
    cp('headline.cvalue.4', (p_r_f_x, 'Оплата', 'услуг'))
    cp('headline.cvalue.5',
       ('порядок и сроки', 'оплаты', 'согласовываются Сторонами в Дополнительных соглашениях к настоящему'))

    cp('headline.pricecond.1', ('УСЛОВИЯ ', 'ПЛАТЕЖЕЙ', ''))
    cp('headline.pricecond.3', ('Условия и порядок', 'расчетов.', ''))
    cp('headline.pricecond.4', (p_r_f_x, 'СТОИМОСТЬ', 'УСЛУГ, ПОРЯДОК ИХ ПРИЕМКИ И РАСЧЕТОВ'))
    cp('headline.pricecond.5', (' АРЕНДНАЯ', 'ПЛАТА', 'ПОРЯДОК ВНЕСЕНИЯ АРЕНДНОЙ ПЛАТЫ'))

    cp('headline.dates.1', (p_r_f_x, 'СРОКИ.', 'ВЫПОЛНЕНИЯ РАБОТ. Порядок выполнения работ.'))

    cp('headline.dates.2', (p_r_f_x, 'СРОК',
                            'ДЕЙСТВИЯ. \n настоящий договор вступает в силу с момента подписания сторонами, изменения и дополнения к договору оформляются письменным соглашением сторон, продленным на каждый последующий год'))

    cp('headline.break', (p_r_f_x, 'Расторжение',
                          'договора. \n досрочное расторжение договора, предупреждением о прекращении, расторгается в случаях, предусмотренных действующим законодательством, в одностороннем порядке'))

    # обязанности сторон
    cp('headline.obl.1', (p_r_f_x, 'права и обязанности', 'сторон.\n'))

    cp('headline.obl.2', (p_r_f_x, 'ОБЯЗАТЕЛЬСТВА', 'сторон.\n'))
    cp('headline.obl.3', (p_r_f_x, 'обязанности', 'сторон'))

    cp('headline.obl.4', (p_r_f_x, 'ГАРАНТИЙНЫЕ', 'ОБЯЗАТЕЛЬСТВА.'))


    cp('headline.resp', (p_r_f_x, 'Ответственность сторон.\n',
                         'невыполнения или ненадлежащего выполнения своих обязательств, несут ответственность в соответствии с действующим законодательством'))

    # ФОРС - МАЖОР
    cp('headline.forcemajor.1', (p_r_f_x, 'НЕПРЕОДОЛИМАЯ СИЛА.', 'ФОРС-МАЖОРНЫЕ ОБСТОЯТЕЛЬСТВА'))
    cp('headline.forcemajor.2', (p_r_f_x, 'ОБСТОЯТЕЛЬСТВА НЕПРЕОДОЛИМОЙ СИЛЫ', ''))

    # КОНФИДЕНЦИАЛЬНОСТЬ
    cp('headline.security.1', (p_r_f_x, 'КОНФИДЕНЦИАЛЬНОСТЬ ИНФОРМАЦИИ.', ''))
    cp('headline.security.2', (p_r_f_x, 'КОНФИДЕНЦИАЛЬНОСТЬ', ''))


    cp('headline.special.1', (p_r_f_x + 'ОСОБЫЕ, дополнительные', ' УСЛОВИЯ.', ''))
    cp('headline.special.2', (p_r_f_x, 'ЗАКЛЮЧИТЕЛЬНЫЕ ПОЛОЖЕНИЯ.', ''))

    cp('headline.appl', (p_r_f_x, 'ПРИЛОЖЕНИЯ', 'К ДОГОВОРУ'))

    # РЕКВИЗИТЫ СТОРОН
    cp('headline.addresses.1', (p_r_f_x, 'РЕКВИЗИТЫ СТОРОН', 'ЮРИДИЧЕСКИЕ АДРЕСА'))
    cp('headline.addresses.2', (p_r_f_x, 'ЮРИДИЧЕСКИЕ АДРЕСА', 'РЕКВИЗИТЫ СТОРОН'))

    cp('headline.conficts', (p_r_f_x, 'Споры и разногласия.', ''))

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

  def _build_subject_patterns(self):

    def cp(a, p, c=None):
      cnt = len(self.patterns)
      if c is None:
        c = ""
      return self.create_pattern(f'x_{subj}.{cnt}', (a, p, c))

    # ----------------------------------------------------------
    subj = ContractSubject.Charity
    if True:
      cp('договор',
         'благотворительного',
         'пожертвования')

      cp('договор о предоставлении',
         'безвозмездной помощи',
         'финансовой')

      cp('Благотворитель обязуется передать',
         'Благополучателю',
         'в порядке добровольного')

      cp('проведение',
         'благотворительных',
         '')

      cp('', 'Благотворитель', '')
      cp('', 'Благополучатель', '')

      cp('принимает в качестве',
         'Пожертвования',
         '')

      cp('',
         'Жертвователь',
         'безвозмездно передает в собственность, а Благополучатель принимает')

      cp('Жертвователь', 'безвозмездно', '')

    # ----------------------------------------------------------
    subj = ContractSubject.Service
    if True:
      # TODO: 🚷 sorry, order matters!!! do not 🚷 touch

      #     cp('ПРОДАВЕЦ обязуется передать в собственность ПОКУПАТЕЛЯ, а', 'ПОКУПАТЕЛЬ', 'обязуется принять и оплатить')

      cp('Исполнитель обязуется своими силами',
         'выполнить работы',
         'по разработке')

      cp('Исполнитель обязуется',
         'оказать услуги',
         '')

      cp('принимает на себя',
         'обязательство разработать',
         '')

      cp('Исполнитель предоставляет ', 'услуги', 'в виде')
      cp('договор на оказание', 'платных', 'услуг')
      cp('Заказчик обязуется оплатить', 'услуги', '')

    # ----------------------------------------------------------
    subj = ContractSubject.RealEstate
    if True:
      cp('Покупатель обязуется', 'оплатить Объект недвижимого', 'имущества')
      cp('Продавец обязуется', 'передать в собственность',
         'Покупателя объект недвижимого имущества здание, расположенное по адресу')
      cp('Продавец обязуется передать в собственность', 'недвижимое имущество', 'объекты , земельные участки')

    subj = ContractSubject.Loans
    if True:
      cp('Займодавец передает Заемщику', 'в качестве займа', 'денежные средства')
      cp('', 'Займодавец передает Заемщику', 'в качестве займа денежные средства')
      cp('Займодавец передает Заемщику', 'целевой заем', 'траншами на возобновляемой основе')
      cp('Заемщик обязуется вернуть указанную', 'сумму займа', 'вместе с причитающимися процентами')

    subj = ContractSubject.PledgeEncumbrance
    if True:
      cp('Залогодержатель принимает, а', 'Залогодатель передает в залог', 'в качестве обеспечения')

    subj = ContractSubject.Renting
    if True:
      cp('Арендодатель передает, а', 'Арендатор принимает в аренду', '(во временное владение и пользование) здание')
      cp('', 'Арендодатель',
         'обязуется передать Арендатору во временное владение и пользование (аренду) недвижимое имущество')

    subj = ContractSubject.AgencyContract
    if True:
      cp('', 'Агент', 'обязуется совершать по поручению Принципала')


if __name__ == '__main__':
  CPF = ContractPatternFactory(ElmoEmbedder.get_instance('elmo'))
  for p in CPF.patterns:
    print(p.prefix_pattern_suffix_tuple)
