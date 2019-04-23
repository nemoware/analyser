from legal_docs import rectifyed_sum_by_pattern_prefix
from ml_tools import max_exclusive_pattern_by_prefix, momentum, cut_above, relu
from patterns import AbstractPatternFactoryLowCase


class ContractPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)
    # self.headlines = ['subj', 'contract', 'def', 'price.', 'pricecond', 'terms', 'dates', 'break', 'rights', 'obl',
    #                   'resp', 'forcemajor', 'confidence', 'special', 'appl', 'addresses', 'conficts']

    self.headlines = ['subj', 'contract', 'price.', 'pricecond', 'dates',
                      'resp', 'forcemajor', 'confidence', 'appl', 'addresses', 'conficts']

    self._build_head_patterns()
    self._build_sum_patterns()
    self._build_subject_patterns()
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
