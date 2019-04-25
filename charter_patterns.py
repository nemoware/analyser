from typing import List

from legal_docs import org_types, make_soft_attention_vector, CharterDocument, deprecated, \
  rectifyed_sum_by_pattern_prefix
from ml_tools import cut_above, relu, momentum
from ml_tools import filter_values_by_key_prefix
from patterns import AbstractPatternFactoryLowCase, PatternSearchResult
from structures import ContractSubject
from parsing import known_subjects


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
    build_lawsuit_patterns(self)
    _build_realestate_patterns(self)
    # _build_deal_patterns(self)

    _build_margin_values_patterns(self)

    for subj in known_subjects:
      if subj is not ContractSubject.Other:
        pb = filter_values_by_key_prefix(self.patterns_dict, f'x_{subj}')
        assert len(pb) > 0, subj

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


def build_charity_patterns(factory):
  def cp(a, p, c=None):
    cnt = len(factory.patterns)
    if c is None:
      c = ""
    return factory.create_pattern(f'x_{ContractSubject.Charity}.{cnt}', (a, p, c))

  cp('договор',
     p='благотворительного',
     c='пожертвования')

  cp('одобрение внесения Обществом каких-либо вкладов или пожертвований на политические или',
     p='благотворительные',
     c='цели')

  cp('одобрение внесения Обществом каких-либо вкладов или',
     p='пожертвований',
     c='на политические или благотворительные цели ')

  cp('предоставление',
     p='безвозмездной',
     c='помощи финансовой')

  cp('согласование сделок',
     'дарения')


def build_lawsuit_patterns(factory):
  def cp(a, p, c=None):
    cnt = len(factory.patterns)
    if c is None:
      c = ""
    return factory.create_pattern(f'x_{ContractSubject.Lawsuit}.{cnt}', (a, p, c))

  cp('начало/урегулирование любых',
     'судебных',
     'споров, подписание мирового соглашения, признание иска, отказ от иска, а также любые другие ')

  cp('судебных споров, цена ',
     'иска',
     'по которым превышает')


def _build_margin_values_patterns(factory):
  suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
  prefix = 'совершение сделок '

  def cp(a, p, c=None):
    cnt = len(factory.patterns)
    if c is None:
      c = ""
    return factory.create_pattern(f'margin_value.{cnt}', (a, p, c))

  # less than
  cp(prefix + 'стоимость не более ', '0', suffix)
  cp(prefix + 'цена не больше ', '0', suffix)
  cp(prefix + 'стоимость', '< 0', suffix)
  cp(prefix + 'цена менее', '0', suffix)
  cp(prefix + 'цена ниже', '0', suffix)
  cp(prefix + 'стоимость не может превышать ', '0', suffix)
  cp(prefix + 'лимит соглашения', '0', suffix)
  cp(prefix + 'верхний лимит стоимости', '0', suffix)
  cp(prefix, 'максимум 0', suffix)
  cp(prefix, 'до 0', suffix)
  cp(prefix, 'но не превышающую 0', suffix)
  cp(prefix, 'совокупное пороговое значение 0', suffix)

  # greather than
  cp(prefix + 'составляет', 'более 0', suffix)
  cp(prefix + 'превышает', ' 0', suffix)
  cp(prefix + 'свыше', ' 0', suffix)
  cp(prefix + 'сделка имеет стоимость, равную или превышающую', ' 0', suffix)


def _build_realestate_patterns(factory):
  def cp(a, p, c=None):
    cnt = len(factory.patterns)
    if c is None:
      c = ""
    return factory.create_pattern(f'x_{ContractSubject.RealEstate}.{cnt}', (a, p, c))

  cp('принятие решений о совершении сделок c ', 'имуществом', '')
  cp('отчуждению активов общества ( включая', 'недвижимость', '', )


def find_sentences_by_pattern_prefix(factory, head_sections: dict, pattern_prefix) -> dict:
  quotes_by_head_type = {}
  for section_name in head_sections:
    subdoc = head_sections[section_name].body
    bounds: List[PatternSearchResult] = subdoc.find_sentences_by_pattern_prefix(factory, pattern_prefix)
    quotes_by_head_type[section_name] = bounds

  return quotes_by_head_type


@deprecated
def make_constraints_attention_vectors(subdoc):
  # TODO: move to notebook, too much tuning
  value_attention_vector, _c1 = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, 'sum_max',
                                                                relu_th=0.4)
  value_attention_vector = cut_above(value_attention_vector, 1)
  value_attention_vector = relu(value_attention_vector, 0.6)
  value_attention_vector = momentum(value_attention_vector, 0.7)

  deal_attention_vector, _c2 = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, 'd_order',
                                                               relu_th=0.5)
  deal_attention_vector = cut_above(deal_attention_vector, 1)
  deal_attention_vector = momentum(deal_attention_vector, 0.993)

  margin_attention_vector, _c3 = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, 'sum__',
                                                                 relu_th=0.5)
  margin_attention_vector = cut_above(margin_attention_vector, 1)
  margin_attention_vector = momentum(margin_attention_vector, 0.95)
  margin_attention_vector = relu(margin_attention_vector, 0.65)

  margin_value_attention_vector = relu((margin_attention_vector + value_attention_vector) / 2, 0.6)

  deal_value_attention_vector = (deal_attention_vector + margin_value_attention_vector) / 2
  deal_value_attention_vector = relu(deal_value_attention_vector, 0.75)

  return {
    'value_attention_vector': value_attention_vector,
    'deal_attention_vector': deal_attention_vector,
    'margin_attention_vector': margin_attention_vector,
    'margin_value_attention_vector': margin_value_attention_vector,

    'deal_value_attention_vector': deal_value_attention_vector
  }
