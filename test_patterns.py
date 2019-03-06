import unittest

from patterns import *
import numpy as np


class CoumpoundFuzzyPatternTestCase(unittest.TestCase):

    def test_onehot(self):
        ep = ExclusivePattern()
        a = np.array([[3.0, 2, 3], [2, 3, 5]])
        mask=-np.inf
        m = ep.onehot_column(a, -np.inf)
        print(m)
        self.assertTrue(np.allclose(m, np.array([[3, mask, mask], [mask, 3, 5]])))

    def test_tokenize_doc(self):
        doc = LegalDocument()
        tokens = doc.tokenize('aa bb cc')
        print (tokens)
        self.assertEqual(3 + TEXT_PADDING + 1, len(tokens))

    def test_tokenize_doc_custom_padding(self):
        doc = LegalDocument()
        padding = 0
        tokens = doc.tokenize('aa bb cc', padding)
        print (tokens)
        self.assertEqual(3 + padding + 1, len(tokens))

    def test_eval_distances_soft_pattern(self):
        point1 = [1, 3]
        point2 = [1, 7]

        point3 = [1, 6]
        point35 = [1, 6.5]

        fp1 = FuzzyPattern(np.array([[point3], [point35]]))

        text_emb = np.array([point1, point2, point3])
        sums = fp1._eval_distances(text_emb)
        self.assertEqual(len(text_emb), len(sums))

        line0 = sums[:, 0]

        self.assertGreater(line0[1], line0[2])
        self.assertGreater(line0[0], line0[2])

    def test_eval_distances_soft_pattern2(self):
        point1 = [1, 3]
        point2 = [1, 7]

        point3 = [1, 6]
        point35 = [1, 6.5]

        fp1 = FuzzyPattern(np.array([[point3], [point35]]))

        text_emb = np.array([point1, point3, point2, point2])
        sums = fp1._eval_distances(text_emb)
        self.assertEqual(len(text_emb), len(sums))

        line0 = sums[:, 0]
        print(line0)

        self.assertGreater(line0[2], line0[1])
        self.assertGreater(line0[0], line0[1])
        self.assertGreater(line0[3], line0[1])

    def test_eval_distances(self):
        point1 = [1, 3]
        point2 = [1, 7]

        point3 = [1, 6]

        fp1 = FuzzyPattern(np.array([[point3], [point2]]))

        text_emb = np.array([point1, point2, point3])
        sums = fp1._eval_distances(text_emb)
        self.assertEqual(len(text_emb), len(sums))

        line0 = sums[:, 0]
        # print(line0)
        # print(sums[:,1])

        self.assertAlmostEqual(line0[2], 0)

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
