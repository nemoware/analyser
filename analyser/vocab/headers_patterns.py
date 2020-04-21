import numpy as np

from analyser.patterns import CentroidPattensBuilder

headers_patterns = {
  'cvalue': [
    'цена договора',
    'СТОИМОСТЬ РАБОТ',
    'Расчеты по договору',
    'Оплата услуг',
    'АРЕНДНАЯ ПЛАТА'],

  'pricecond': [
    'порядок и сроки оплаты',
    'УСЛОВИЯ ПЛАТЕЖЕЙ',
    'Условия и порядок расчетов',
    'ПОРЯДОК ПРИЕМКИ И РАСЧЕТОВ',
    'ПОРЯДОК ВНЕСЕНИЯ АРЕНДНОЙ ПЛАТЫ'
  ],

  'obl': [
    'права и обязанности сторон',
    'Обязательства сторон',
    'обязанности сторон',
    'ГАРАНТИЙНЫЕ ОБЯЗАТЕЛЬСТВА'
  ],

  'forcemajor': [
    'НЕПРЕОДОЛИМАЯ СИЛА',
    'ФОРС-МАЖОРНЫЕ ОБСТОЯТЕЛЬСТВА',
    'ОБСТОЯТЕЛЬСТВА НЕПРЕОДОЛИМОЙ СИЛЫ',
    'ФОРС-МАЖОР'
  ],

  'security': [
    'КОНФИДЕНЦИАЛЬНОСТЬ ИНФОРМАЦИИ',
    'КОНФИДЕНЦИАЛЬНОСТЬ'
  ],

  'addresses': [
    'РЕКВИЗИТЫ СТОРОН',
    'ЮРИДИЧЕСКИЕ АДРЕСА',
    'АДРЕСА И РЕКВИЗИТЫ СТОРОН'
  ],

  'subj': [
    'ПРЕДМЕТ ДОГОВОРА'
  ],

  'def': [
    'Термины и определения',
    'толкования'
  ],

  'dates': [
    'СРОК ДЕЙСТВИЯ',
    'СРОК ДЕЙСТВИЯ ДОГОВОРА',
    'СРОКИ ВЫПОЛНЕНИЯ РАБОТ',
    'СРОКИ.'
  ],

  'contract': [
    'ДОГОВОР'
  ]

}


def _normalize_patterns():
  for k in headers_patterns:
    arr = headers_patterns[k]
    alternatives = []
    for phrase in arr:
      alternatives += [phrase, phrase.capitalize(), phrase.lower(), phrase.upper(), phrase.upper() + '.',
                       phrase.upper() + '\n']

    headers_patterns[k] = np.unique(alternatives)


_normalize_patterns()

if __name__ == '__main__':
  # _normalize_patterns()
  print(headers_patterns)

  cp = CentroidPattensBuilder()
  cp.calc_patterns_centroids(headers_patterns)
  pass
