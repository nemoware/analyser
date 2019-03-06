import unittest

from patterns import *
import numpy as np


class CoumpoundFuzzyPatternTestCase(unittest.TestCase):

    def test_eval_distances(self):
        point1 = [1, 3]
        point2 = [1, 7]

        point3 = [1, 6]

        fp1 = FuzzyPattern(np.array([[point3], [point2]]))

        text_emb = np.array([point1, point2, point3])
        sums = fp1._eval_distances(text_emb)
        self.assertEqual(len (text_emb), len(sums))

        line0 = sums[:,0]
        # print(line0)
        # print(sums[:,1])

        self.assertAlmostEqual(line0[2],0)

        self.assertGreater(line0[0], line0[1])
        self.assertGreater(line0[1], line0[2])
        self.assertGreater(line0[0], line0[2])

    def test_coumpound_find(self):
        point1 = [1, 3]
        point2 = [1, 7]
        point3 = [1, 6]

        fp2 = FuzzyPattern(np.array([[point2]]))
        # fp2 = FuzzyPattern(np.array([[point2]]))

        cp = CoumpoundFuzzyPattern()
        cp.add_pattern(fp2, 0.5)
        # cp.add_pattern(fp2, 2)

        text_emb = np.array([point1, point2, point3])
        min_i, sums, confidence = cp.find(text_emb, text_right_padding=0)

        print(min_i, sums, confidence)

        self.assertEqual(1, min_i)


if __name__ == '__main__':
    unittest.main()
