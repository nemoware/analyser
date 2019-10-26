import unittest

from text_tools import find_best_sentence_end


class TextToolsTestCase(unittest.TestCase):

  def test_find_best_sentence_end_0(self):
    x = 'ие? д'
    be = find_best_sentence_end(x)
    self.assertEqual("ие?", x[:be])

  def test_find_best_sentence_end_1(self):
    x = 'ие. д'
    be = find_best_sentence_end(x)
    self.assertEqual("ие.", x[:be])

  def test_find_best_sentence_end_2(self):
    x = 'ие) д'
    be = find_best_sentence_end(x)
    self.assertEqual("ие)", x[:be])

  def test_find_best_sentence_end_3(self):
    x = 'ие) д!'
    be = find_best_sentence_end(x)
    self.assertEqual("ие) д!", x[:be])

  def test_find_best_sentence_end_4(self):
    x = 'иед'
    be = find_best_sentence_end(x)
    self.assertEqual("иед", x[:be])

  def test_find_best_sentence_end_5(self):
    x = 'иед: нет'
    be = find_best_sentence_end(x)
    self.assertEqual("иед:", x[:be])


if __name__ == '__main__':
  unittest.main()
