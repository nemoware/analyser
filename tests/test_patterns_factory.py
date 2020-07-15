import unittest

from analyser.patterns import AbstractPatternFactory
from tests.test_utilits import FakeEmbedder


class FakePatternFactory(AbstractPatternFactory):
  pass
  # def __init__(self, embedder):
  #     super(embedder)


class CoumpoundFuzzyPatternTestCase(unittest.TestCase):

  # def __init__(self):

  def test_embedder(self):
    point1 = [1, 6, 4]

    PF = AbstractPatternFactory()

    fp1 = PF.create_pattern('p1', ('prefix', 'pat 2', 'suffix'))
    fp2 = PF.create_pattern('p2', ('prefix', 'pat', 'suffix 2'))
    fp3 = PF.create_pattern('p3', ('', 'a b c', ''))

    self.assertEqual(3, len(PF.patterns))

    PF.embedd(FakeEmbedder(point1))

    self.assertEqual(2, len(fp1.embeddings))
    self.assertEqual(1, len(fp2.embeddings))
    self.assertEqual(3, len(fp3.embeddings))


if __name__ == '__main__':
  unittest.main()
