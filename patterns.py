from text_normalize import *
from text_tools import *

TEXT_PADDING = 10  # maximum pattern len (in words)
TEXT_PADDING_SYMBOL = ' '
# DIST_FUNC = dist_frechet_cosine_undirected
DIST_FUNC = dist_mean_cosine
# DIST_FUNC = dist_cosine_housedorff_undirected
PATTERN_THRESHOLD = 0.75  # 0...1


class FuzzyPattern:

    def __init__(self, patterns_embeddings_list, _name='undefined'):
        assert patterns_embeddings_list[0][0][0]
        self.patterns_embeddings_list = patterns_embeddings_list
        self.name = _name
        self.pattern_text = None
        self.soft_sliding_window_borders = False

    def __str__(self):
        return ' '.join(['FuzzyPattern:', str(self.name), str(self.pattern_text)])

    def _eval_distances(self, _text, dist_function=DIST_FUNC, whd_padding=0, wnd_mult=1):
        """
          For each token in the given sentences, it calculates the semantic distance to
          each and every pattern in _pattens arg.

          WARNING: may return None!

          TODO: adjust sliding window size
        """

        _distances = np.zeros((len(_text), len(self.patterns_embeddings_list)))

        for pattern_index in range(len(self.patterns_embeddings_list)):

            _pat = self.patterns_embeddings_list[pattern_index]

            window_size = wnd_mult * len(_pat) + whd_padding
            if window_size > len(_text):
                print('ERROR: window_size > len(_text)', window_size, '>', len(_text))
                return None

            for word_index in range(0, len(_text) - window_size + 1):
                _fragment = _text[word_index: word_index + window_size]
                _distances[word_index, pattern_index] = dist_function(_fragment, _pat)

        return _distances

    def _eval_distances_multi_window(self, _text, dist_function=DIST_FUNC):
        distances = []
        distances.append(self._eval_distances(_text, dist_function, whd_padding=0, wnd_mult=1))
        if self.soft_sliding_window_borders:
            distances.append (self._eval_distances(_text, dist_function, whd_padding=2, wnd_mult=1))
            distances.append(self._eval_distances(_text, dist_function, whd_padding=1, wnd_mult=2))
            distances.append(self._eval_distances(_text, dist_function, whd_padding=7, wnd_mult=0))



        sum = None
        cnt = 0
        for d in distances:
            if d is not None:
                cnt = cnt + 1
                if sum is None:
                    sum=np.array(d)
                else:
                    sum += d


        assert cnt > 0
        sum = sum / cnt

        return sum

    def _find_patterns(self, text_ebd):
        """
          text_ebd:  tensor of embeedings
        """
        w_starts = self._eval_distances_multi_window(text_ebd)
        sums_starts = w_starts.sum(1)  # XXX: 'sum' is the AND case, implement also OR -- use 'max'

        return sums_starts

    def find(self, text_ebd, text_right_padding):
        """
          text_ebd:  tensor of embeedings
        """

        sums = self._find_patterns(text_ebd)
        min_i = min_index(sums[:-text_right_padding])  # index of the word with minimum distance to the pattern

        return min_i, sums


class CoumpoundFuzzyPattern:

    def __init__(self):
        self.patterns = {}

    def add_pattern(self, pat, weight=1.0):
        self.patterns[pat] = weight


    def find(self, text_ebd,  text_right_padding):
        assert len(text_ebd) > text_right_padding
        sums = np.zeros(len(text_ebd))

        for p in self.patterns:
            weight = self.patterns[p]
            sp = p._find_patterns(text_ebd)

            sums += sp * weight

        # norm
        sums /= len(self.patterns)
        meaninful_sums = sums
        if text_right_padding > 0:
            meaninful_sums = sums[:-text_right_padding]

        min_i = min_index(meaninful_sums)
        min = sums[min_i]
        mean = meaninful_sums.mean()
        # confidence = sums[min_i] / mean
        sandard_deviation = np.std(meaninful_sums)
        deviation_from_mean = abs(min-mean)
        confidence = sandard_deviation / deviation_from_mean
        return min_i, sums, confidence


class LegalDocument:
    def __init__(self):
        self.original_text = None
        self.filename = None
        self.tokens = None
        self.embeddings = None
        self.normal_text = None

    def normalize_sentences_bounds(self, text):
        sents = nltk.sent_tokenize(text)
        res = ''
        for s in sents:
            a = s.replace('\n', ' ')
            res += a
            res += '\n'

        return res

    def preprocess_text(self, text):
        a = text
        #     a = remove_empty_lines(text)
        a = normalize_text(a, replacements_regex)
        a = self.normalize_sentences_bounds(a)

        return a

    def read(self, name):
        print("reading...", name)
        self.filename = name
        txt = ""
        with open(name, 'r') as f:
            self.original_text = f.read()

    def tokenize(self, _txt=None, padding = TEXT_PADDING):
        if _txt is None: _txt = self.normal_text

        _words = tokenize_text(_txt)

        sparse_words = []
        end = len(_words)
        last_cr_index = 0
        for i in range(end):
            if (_words[i] == '\n') or i == end - 1:
                chunk = _words[last_cr_index:i + 1]
                chunk.extend([TEXT_PADDING_SYMBOL] * padding)
                sparse_words += chunk
                last_cr_index = i + 1

        return sparse_words

    def parse(self, txt=None, padding = TEXT_PADDING):
        if txt is None: txt = self.original_text
        self.normal_text = self.preprocess_text(txt)

        self.tokens = self.tokenize(padding = padding)
        return self.tokens
        # print('TOKENS:', self.tokens[0:20])

    def embedd(self, pattern_factory):
        self.embeddings, _wrds = pattern_factory.embedder.embedd_tokenized_text(self.tokens)
        self.embeddings = self.embeddings[0]


class AbstractPatternFactory:
    CONFIDENCE_DISTANCE = 0.25  # TODO: must depend on distance function

    def __init__(self, embedder):
        self.embedder = embedder
        self.patterns = None

    def _embedd(self, p):
        arr = []
        for k, v in p.items():
            arr.append([k, v])

        slice = [arr[i][1:2][0] for i in range(len(arr))]

        # =========
        patterns_emb = self.embedder.embedd_contextualized_patterns(slice)
        # =========

        self.patterns = {}
        for i in range(len(patterns_emb)):
            name = arr[i][0]
            fp = FuzzyPattern([patterns_emb[i]], name)
            fp.pattern_text = arr[i][1]
            self.patterns[name] = fp

        return self.patterns
