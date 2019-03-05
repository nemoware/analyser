from text_tools import *

TEXT_PADDING = 10  # maximum pattern len (in words)
TEXT_PADDING_SYMBOL = ' '
# DIST_FUNC = dist_frechet_cosine_undirected
DIST_FUNC = dist_mean_cosine
# DIST_FUNC = dist_cosine_housedorff_undirected
PATTERN_THRESHOLD = 0.75  # 0...1


class FuzzyPattern:

    def __init__(self, start_tokens_emb, _name='undefined'):
        self.start = start_tokens_emb
        self.name = _name
        self.pattern_text = None


    def __str__(self):
        return ' '.join(['FuzzyPattern:', str(self.name), str(self.pattern_text)])


    def _eval_distances(self, _text, _patterns, dist_function=DIST_FUNC, whd_padding=2, wnd_mult=1):
        """
          For each token in the given sentences, it calcultes the semantic distance to
          each and every pattern in _pattens arg.

          TODO: adjust sliding window size
        """

        _distances = np.zeros((len(_text), len(_patterns)))

        for j in range(0, len(_patterns)):

            _pat = _patterns[j]

            window_size = wnd_mult * len(_pat) + whd_padding

            for i in range(0, len(_text) - TEXT_PADDING):
                _fragment = _text[i: i + window_size]
                _distances[i, j] = dist_function(_fragment, _pat)

        return _distances

    def _eval_distances_multi_window(self, _text, _patterns, dist_function=DIST_FUNC):
        d1 = self._eval_distances(_text, _patterns, dist_function, whd_padding=2, wnd_mult=1)
        d2 = self._eval_distances(_text, _patterns, dist_function, whd_padding=1, wnd_mult=2)
        d3 = self._eval_distances(_text, _patterns, dist_function, whd_padding=7, wnd_mult=0)
        d4 = self._eval_distances(_text, _patterns, dist_function, whd_padding=0, wnd_mult=1)

        sum = (d1 + d2 + d3 + d4) / 4

        return sum

    def _find_patterns(self, text_ebd, threshold=PATTERN_THRESHOLD):
        """
          text_ebd:  tensor of embeedings
        """
        w_starts = self._eval_distances_multi_window(text_ebd, self.start)
        sums_starts = w_starts.sum(1)  # XXX: 'sum' is the AND case, implement also OR -- use 'max'

        return sums_starts

    def find(self, text_ebd, threshold=PATTERN_THRESHOLD, text_right_padding=TEXT_PADDING):
        """
          text_ebd:  tensor of embeedings
        """

        sums = self._find_patterns(text_ebd, threshold)
        min_i = min_index(sums[:-text_right_padding])  # index of the word with minimum distance to the pattern

        return min_i, sums


class CoumpoundFuzzyPattern:

    def __init__(self):
        self.patterns = {}

    def add_pattern(self, pat, weight=1.0):
        self.patterns[pat] = weight

    def find(self, text_ebd, threshold=PATTERN_THRESHOLD, text_right_padding=TEXT_PADDING):
        sums = np.zeros(len(text_ebd))
        for p in self.patterns:
            sp = p._find_patterns(text_ebd, threshold)

            sums += sp * self.patterns[p]

        sums /= len(self.patterns)
        meaninful_sums = sums[:-text_right_padding]
        min_i = min_index(meaninful_sums)
        mean = meaninful_sums.mean()
        confidence = sums[min_i] / mean
        return min_i, sums, confidence
