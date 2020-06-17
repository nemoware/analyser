import unittest

from analyser.documents import split_sentences_into_map
from analyser.hyperparams import HyperParameters
from analyser.text_tools import find_best_sentence_end, split_into_sentences, find_top_spans


class TextToolsTestCase(unittest.TestCase):

  def test_find_top_spans_no_resuslts(self):
    v = [0.]
    topspans = find_top_spans(v, limit=100)
    print(topspans)
    self.assertEqual(0, len(topspans))

  def test_find_top_spans_empty(self):
    v = []
    topspans = find_top_spans(v, limit=100)
    print(topspans)
    self.assertEqual(0, len(topspans))

  def test_find_top_spans(self):
    v = [0., 1., 0.6, 0., 0.]
    topspans = find_top_spans(v, limit=1)
    print(topspans)
    self.assertEqual(1, len(topspans))

  def test_find_top_spans_gap(self):
    GAP = 0.0
    v = [GAP, 0.9, 0.6, GAP, GAP, 1., GAP, GAP, GAP]
    print(v)
    topspans = find_top_spans(v, maxgap=1)
    self.assertEqual(2, len(topspans))

    topspans = find_top_spans(v, maxgap=3)
    self.assertEqual(1, len(topspans))

  def test_find_top_spans_limit(self):
    v = [0., 0.9, 0.6, 0., 0., 0., 0., 0., 1.]
    topspans = find_top_spans(v, limit=1)
    print(topspans)
    self.assertEqual(1, len(topspans))
    sp: slice = topspans[0]
    self.assertEqual(8, sp.start)
    self.assertEqual(9, sp.stop)

  def test_find_top_spans_no_limit(self):
    v = [0., 0.9, 0.6, 0., 0., 0., 0., 0., 1.]
    topspans = find_top_spans(v)
    print(topspans)
    self.assertEqual(2, len(topspans))
    sp: slice = topspans[0]
    self.assertEqual(8, sp.start)
    self.assertEqual(9, sp.stop)

    sp: slice = topspans[1]
    self.assertEqual(1, sp.start)
    self.assertEqual(3, sp.stop)

  def test_split_into_sentences(self):
    s = 'Предложение 1.Предложение 2.'
    spans = split_into_sentences(s, 20)

    self.assertEqual('Предложение 2.', s[spans[1][0]:spans[1][1]])
    self.assertEqual('Предложение 1.', s[spans[0][0]:spans[0][1]])

  def test_split_into_sentences_0(self):
    s = 'Предложение 1. Предложение 2.'
    spans = split_into_sentences(s, 20)

    self.assertEqual(' Предложение 2.', s[spans[1][0]:spans[1][1]])
    self.assertEqual('Предложение 1.', s[spans[0][0]:spans[0][1]])

  def test_find_best_sentence_end_0(self):
    x = 'ие? д'
    be, char = find_best_sentence_end(x)
    self.assertEqual("ие?", x[:be])

  def test_find_best_sentence_end_1(self):
    x = 'ие. д'
    be, char = find_best_sentence_end(x)
    self.assertEqual("ие.", x[:be])

  def test_find_best_sentence_end_2(self):
    x = 'ие) д'
    be, char = find_best_sentence_end(x)
    self.assertEqual("ие)", x[:be])

  def test_find_best_sentence_end_3(self):
    x = 'ие) д!'
    be, char = find_best_sentence_end(x)
    self.assertEqual("ие) д!", x[:be])

  def test_find_best_sentence_end_4(self):
    x = 'иед'
    be, _ = find_best_sentence_end(x)
    self.assertEqual("иед", x[:be])

  def test_find_best_sentence_end_5(self):
    x = 'иед: нет'
    be, char = find_best_sentence_end(x)
    print('char=', x[be - 1], char)
    self.assertEqual("иед:", x[:be])

  def test_split_sentences_into_map(self):
    tt = 'принятие решения о согласии на совершение крупной сделки, связанной с приобретением, ' \
         'отчуждением или возможностью отчуждения Обществом прямо или косвенно имущества, цена или ' \
         'балансовая стоимость которого без учета НДС составляет 50 и более процентов балансовой стоимости ' \
         'активов Общества, определенной по данным его бухгалтерской (финансовой) отчетности на последнюю ' \
         'отчетную дату, либо крупной сделки, предусматривающей обязанность Общества передать имущество ' \
         'во временное владение и (или) пользование либо предоставить третьему лицу право использования ' \
         'результата интеллектуальной деятельности или средства индивидуализации на условиях лицензии, ' \
         'если их балансовая стоимость без учета НДС составляет 25 и более процентов балансовой стоимости ' \
         'активов Общества, определенной по данным его бухгалтерской (финансовой) отчетности на последнюю отчетную дату'

    tm = split_sentences_into_map(tt, HyperParameters.charter_sentence_max_len)

    for token in tm.tokens:
      print(len(token), token)


if __name__ == '__main__':
  unittest.main()
