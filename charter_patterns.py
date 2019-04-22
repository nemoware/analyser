from charity_finder import build_charity_patterns
from legal_docs import org_types, make_soft_attention_vector, CharterDocument
from patterns import AbstractPatternFactoryLowCase


class CharterPatternFactory(AbstractPatternFactoryLowCase):

  def __init__(self, embedder):
    AbstractPatternFactoryLowCase.__init__(self, embedder)

    self.headlines = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie', 'name']

    self._build_head_patterns()

    self._build_order_patterns()
    self._build_sum_margin_extraction_patterns()
    self._build_sum_patterns()

    self._build_ner_patterns()

    build_charity_patterns(self)

    self.embedd()

  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    head_prfx = "\n"

    cp('competence', ('', 'компетенция', ''))

    cp('headline.name.1', ('Полное', 'фирменное наименование', 'общества на русском языке:'))
    cp('headline.name.2', ('', 'ОБЩИЕ ПОЛОЖЕНИЯ', ''))
    cp('headline.name.3', ('', 'фирменное', ''))
    cp('headline.name.4', ('', 'русском', ''))
    cp('headline.name.5', ('', 'языке', ''))
    cp('headline.name.6', ('', 'полное', ''))

    #     cp('headline.head.all.comp.a', (head_prfx, 'компетенции', 'общего собрания акционеров общества'))
    #     cp('headline.head.all.comp.p', (head_prfx, 'компетенции', 'общего собрания участников общества'))
    cp('headline.head.all.n.a', ('', 'общее собрание акционеров', ''))
    cp('headline.head.all.n.p', ('', 'общее собрание участников', ''))

    #     cp('headline.head.all.4', ('', 'компетенции', ''))
    #     cp('headline.head.all.5', ('', 'собрания', ''))
    #     cp('headline.head.all.6', ('', 'участников', ''))
    #     cp('headline.head.all.7', ('', 'акционеров', ''))

    #     cp('headline.head.directors.comp', (head_prfx, 'компетенция', 'совета директоров общества. К компетенции Совета директоров относятся следующие вопросы'))
    cp('headline.head.directors.n', ('', 'совет директоров общества', ''))
    #     cp('headline.head.directors.3', ('', 'компетенции', ''))
    #     cp('headline.head.directors.4', ('', 'совета', ''))
    #     cp('headline.head.directors.5', ('', 'директоров', ''))

    #     cp('headline.head.pravlenie.comp', ('', 'компетенции', 'правления'))
    cp('headline.head.pravlenie.n', ('', 'правление общества', ''))
    #     cp('headline.head.pravlenie.2', ('', 'компетенции', ''))

    #     cp('headline.head.gen.1', ('', 'компетенции', 'генерального директора'))
    cp('headline.head.gen.2', ('', 'генеральный директор', ''))

  def _build_sum_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'решений о совершении сделок '

    cp('currency', (prefix + 'стоимость', '0', suffix))

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

  def _build_order_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    cp('d_order_consent', ('Порядок', 'одобрения сделок', 'в совершении которых имеется заинтересованность'))
    cp('d_order_solution', ('', 'принятие решений', 'о совершении сделок'))
    cp('d_order_consent_1', ('', 'одобрение заключения', 'изменения или расторжения какой-либо сделки Общества'))

    prefix = 'принятие решения о согласии на совершение или о последующем одобрении'

    cp('d_order_deal.1', (prefix, 'cделки', ', стоимость которой равна или превышает'))
    cp('d_order_deal.2', (prefix, 'cделки', ', стоимость которой составляет менее'))

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
