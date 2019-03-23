from typing import List

from patterns import *
from text_normalize import *
from text_tools import *


def mask_sections(section_name_to_weight_dict, doc):
    mask = np.zeros(len(doc.tokens))

    for name in section_name_to_weight_dict:
        section = find_section_by_caption(name, doc.subdocs)
        print([section.start, section.end])
        mask[section.start:section.end] = section_name_to_weight_dict[name]
    return mask


def normalize(x, out_range=(0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    #     return y
    halflen = int(window_len / 2)
    #     return y[0:len(x)]
    return y[(halflen - 1):-halflen]


def find_section_by_caption(cap, subdocs):
    solution_section = None
    mx = 0;
    for subdoc in subdocs:
        d = subdoc.distances_per_pattern_dict[cap]
        _mx = d.max()
        if _mx > mx:
            solution_section = subdoc
            mx = _mx
    return solution_section


class LegalDocument(EmbeddableText):

    def __init__(self, original_text=None):
        self.original_text = original_text
        self.filename = None
        self.tokens = None
        self.embeddings = None
        self.normal_text = None
        self.distances_per_pattern_dict = None

        self.right_padding = 10

        # subdocs
        self.start = None
        self.end = None

    def find_sum_in_section(self):
        raise Exception('not implemented')

    def find_sentence_beginnings(self, best_indexes):
        return [find_token_before_index(self.tokens, i, '\n', 0) for i in best_indexes]

    def calculate_distances_per_pattern(self, pattern_factory: AbstractPatternFactory, dist_function=DIST_FUNC):
        distances_per_pattern_dict = {}
        for pat in pattern_factory.patterns:
            dists = pat._eval_distances_multi_window(self.embeddings, dist_function)
            if self.right_padding > 0:
                dists = dists[:-self.right_padding]
            # TODO: this inversion must be a part of a dist_function
            dists = 1.0 - dists
            distances_per_pattern_dict[pat.name] = dists
            # print(pat.name)

        self.distances_per_pattern_dict = distances_per_pattern_dict
        return self.distances_per_pattern_dict

    def subdoc(self, start, end):

        assert self.tokens is not None
        assert self.embeddings is not None
        assert self.distances_per_pattern_dict is not None

        klazz = self.__class__
        sub = klazz("REF")
        sub.start = start
        sub.end = end
        sub.right_padding = 0

        sub.embeddings = self.embeddings[start:end]

        sub.distances_per_pattern_dict = {}
        for d in self.distances_per_pattern_dict:
            sub.distances_per_pattern_dict[d] = self.distances_per_pattern_dict[d][start:end]

        sub.tokens = self.tokens[start:end]
        return sub

    def split_into_sections(self, caption_pattern_prefix='p_cap_', relu_th=0.5, soothing_wind_size=22):
        tokens = self.tokens
        if (self.right_padding > 0):
            tokens = self.tokens[:-self.right_padding]
        # l = len(tokens)

        captions = rectifyed_mean_by_pattern_prefix(self.distances_per_pattern_dict, caption_pattern_prefix, relu_th)

        captions = normalize(captions)
        captions = smooth(captions, window_len=soothing_wind_size)

        sections = extremums(captions)
        # print(sections)
        sections_starts = [find_token_before_index(self.tokens, i, '\n', 0) for i in sections]
        # print(sections_starts)
        sections_starts = remove_similar_indexes(sections_starts)
        sections_starts.append(len(tokens))
        # print(sections_starts)

        # RENDER sections
        self.subdocs = []
        for i in range(1, len(sections_starts)):
            s = sections_starts[i - 1]
            e = sections_starts[i]
            subdoc = self.subdoc(s, e)
            self.subdocs.append(subdoc)
            # print('-' * 20)
            # render_color_text(subdoc.tokens, captions[s:e])

        return self.subdocs, captions

    def normalize_sentences_bounds(self, text):
        sents = ru_tokenizer.tokenize(text)
        for s in sents:
            s.replace('\n', ' ')

        return '\n'.join(sents)

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
            self.set_original_text(f.read())

    def set_original_text(self, txt):
        self.original_text = txt
        self.tokens = None
        self.embeddings = None
        self.normal_text = None

    def tokenize(self, _txt=None):
        if _txt is None: _txt = self.normal_text

        _words = tokenize_text(_txt)

        sparse_words = []
        end = len(_words)
        last_cr_index = 0
        for i in range(end):
            if (_words[i] == '\n') or i == end - 1:
                chunk = _words[last_cr_index:i + 1]
                chunk.extend([TEXT_PADDING_SYMBOL] * self.right_padding)
                sparse_words += chunk
                last_cr_index = i + 1

        return sparse_words

    def parse(self, txt=None):
        if txt is None: txt = self.original_text
        self.normal_text = self.preprocess_text(txt)

        self.tokens = self.tokenize()
        return self.tokens
        # print('TOKENS:', self.tokens[0:20])

    def embedd(self, pattern_factory):
        self.embeddings, _g = pattern_factory.embedder.embedd_tokenized_text([self.tokens], [len(self.tokens)])
        self.embeddings = self.embeddings[0]


class LegalDocumentLowCase(LegalDocument):

    def __init__(self, original_text):
        LegalDocument.__init__(self, original_text)

    def preprocess_text(self, text):
        a = text
        #     a = remove_empty_lines(text)
        a = normalize_text(a,
                           dates_regex + abbreviation_regex + fixtures_regex +
                           spaces_regex + syntax_regex + numbers_regex +
                           formatting_regex + tables_regex)

        a = self.normalize_sentences_bounds(a)

        return a.lower()


class ContractDocument(LegalDocumentLowCase):
    def __init__(self, original_text):
        LegalDocumentLowCase.__init__(self, original_text)


def rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0):
    c = 0
    sum = None

    for p in distances_per_pattern_dict:
        if p.startswith(prefix):
            # print(p)
            x = distances_per_pattern_dict[p]
            if sum is None:
                sum = np.zeros(len(x))
            relu = x * (x > relu_th)
            sum += relu
            c += 1
    #   deal/=c
    return sum, c


def mean_by_pattern_prefix(distances_per_pattern_dict, prefix):
    #     print('mean_by_pattern_prefix', prefix, relu_th)
    sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0)
    return normalize(sum)


def rectifyed_normalized_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
    return normalize(rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th))


def rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
    #     print('mean_by_pattern_prefix', prefix, relu_th)
    sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th)
    sum /= c
    return sum


def remove_similar_indexes(indexes, min_section_size=20):
    indexes_zipped = []
    indexes_zipped.append(indexes[0])

    for i in range(1, len(indexes)):
        if indexes[i] - indexes[i - 1] > min_section_size:
            indexes_zipped.append(indexes[i])
    return indexes_zipped


def extremums(x):
    extremums = []
    extremums.append(0)
    for i in range(1, len(x) - 1):
        if x[i - 1] < x[i] > x[i + 1]:
            extremums.append(i)
    return extremums


class BasicContractDocument(LegalDocumentLowCase):

    def __init__(self, original_text=None):
        LegalDocumentLowCase.__init__(self, original_text)

    def get_subject_ranges(self, indexes_zipped, section_indexes: List):

        # res = [None] * len(section_indexes)
        # for sec in section_indexes:
        #     for i in range(len(indexes_zipped) - 1):
        #         if indexes_zipped[i][0] == sec:
        #             range1 = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
        #             res[sec] = range1
        #
        #     if res[sec] is None:
        #         print("WARNING: Section #{} not found!".format(sec))
        #
        # return res

        subj_range = None
        head_range = None
        for i in range(len(indexes_zipped) - 1):
            if indexes_zipped[i][0] == 1:
                subj_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
            if indexes_zipped[i][0] == 0:
                head_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
        if head_range is None:
            print("WARNING: Contract type might be not known!!")
            head_range = range(0, 0)
        if subj_range is None:
            print("WARNING: Contract subject might be not known!!")
            if len(self.tokens) < 80:
                _end = len(self.tokens)
            else:
                _end = 80
            subj_range = range(0, _end)
        return head_range, subj_range

    def find_subject_section(self, pattern_fctry: AbstractPatternFactory, numbers_of_patterns):

        self.split_into_sections(pattern_fctry.paragraph_split_pattern)
        indexes_zipped = self.section_indexes

        head_range, subj_range = self.get_subject_ranges(indexes_zipped, [0, 1])

        distances_per_subj_pattern_, ranges_, winning_patterns = pattern_fctry.subject_patterns.calc_exclusive_distances(
            self.embeddings,
            text_right_padding=0)
        distances_per_pattern_t = distances_per_subj_pattern_[:, subj_range.start:subj_range.stop]

        ranges = [np.nanmin(distances_per_subj_pattern_[:-TEXT_PADDING]),
                  np.nanmax(distances_per_subj_pattern_[:-TEXT_PADDING])]

        weight_per_pat = []
        for row in distances_per_pattern_t:
            weight_per_pat.append(np.nanmin(row))

        print("weight_per_pat", weight_per_pat)

        _ch_r = numbers_of_patterns['charity']
        _co_r = numbers_of_patterns['commerce']

        chariy_slice = weight_per_pat[_ch_r[0]:_ch_r[1]]
        commerce_slice = weight_per_pat[_co_r[0]:_co_r[1]]

        min_charity_index = min_index(chariy_slice)
        min_commerce_index = min_index(commerce_slice)

        print("min_charity_index, min_commerce_index", min_charity_index, min_commerce_index)
        self.per_subject_distances = [
            np.nanmin(chariy_slice),
            np.nanmin(commerce_slice)]

        self.subj_range = subj_range
        self.head_range = head_range

        return ranges, winning_patterns

    # TODO: remove
    def __find_sum(self, pattern_factory):

        min_i, sums_no_padding, confidence = pattern_factory.sum_pattern.find(self.embeddings, self.right_padding)

        self.sums = sums_no_padding
        sums = sums_no_padding[:-TEXT_PADDING]

        meta = {
            'tokens': len(sums),
            'index found': min_i,
            'd-range': (sums.min(), sums.max()),
            'confidence': confidence,
            'mean': sums.mean(),
            'std': np.std(sums),
            'min': sums[min_i],
        }

        start, end = get_sentence_bounds_at_index(min_i, self.tokens)
        sentence_tokens = self.tokens[start + 1:end]

        f, sentence = extract_sum_from_tokens(sentence_tokens)

        self.found_sum = (f, (start, end), sentence, meta)

    #     return

    def analyze(self, pattern_factory):
        self.embedd(pattern_factory)
        self._find_sum(pattern_factory)

        self.subj_ranges, self.winning_subj_patterns = self.find_subject_section(
            pattern_factory, {"charity": [0, 5], "commerce": [5, 5 + 7]})


# SUMS -----------------------------

def extract_sum(sentence: str):
    currency_re = re.compile(r'((^|\s+)(\d+[., ])*\d+)(\s*([(].{0,100}[)]\s*)?(евро|руб|доллар))')
    currency_re_th = re.compile(
        r'((^|\s+)(\d+[., ])*\d+)(\s+(тыс\.|тысяч.{0,2})\s+)(\s*([(].{0,100}[)]\s*)?(евро|руб|доллар))')
    currency_re_mil = re.compile(
        r'((^|\s+)(\d+[., ])*\d+)(\s+(млн\.|миллион.{0,3})\s+)(\s*([(].{0,100}[)]\s*)?(евро|руб|доллар))')

    r = currency_re.findall(sentence)
    f = None
    try:
        number = to_float(r[0][0])
        f = (number, r[0][5])
    except:
        r = currency_re_th.findall(sentence)

        try:
            number = to_float(r[0][0]) * 1000
            f = (number, r[0][5])
        except:
            r = currency_re_mil.findall(sentence)
            try:
                number = to_float(r[0][0]) * 1000000
                f = (number, r[0][5])
            except:
                pass

    return f


def extract_sum_from_tokens(sentence_tokens: List):
    sentence = untokenize(sentence_tokens).lower().strip()
    f = extract_sum(sentence)
    return f, sentence


def _extract_sum_from_distances(doc, sums_no_padding):
    max_i = np.argmax(sums_no_padding)
    start, end = get_sentence_bounds_at_index(max_i, doc.tokens)
    sentence_tokens = doc.tokens[start + 1:end]

    f, sentence = extract_sum_from_tokens(sentence_tokens)

    return (f, (start, end), sentence)


def extract_sum_from_doc(doc: LegalDocument, mask=None):
    sum_pos, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max')
    sum_neg, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max_neg')

    sum_pos -= sum_neg

    if mask is not None:
        sum_pos *= mask

    #   sum_ctx = smooth(sum_ctx, window_len=10)

    sum_dist = normalize(sum_pos)

    # render_color_text(doc.tokens, sum_dist, print_debug=True)

    x = _extract_sum_from_distances(doc, sum_dist)
    return x


class ProtocolDocument(LegalDocumentLowCase):

    def __init__(self, original_text=None):
        LegalDocumentLowCase.__init__(self, original_text)

    def make_solutions_mask(self):

        section_name_to_weight_dict = {}
        for i in range(1, 5):
            cap = 'p_cap_solution{}'.format(i)
            section_name_to_weight_dict[cap] = 0.35

        mask = mask_sections(section_name_to_weight_dict, self)
        mask += 0.5
        if self.right_padding > 0:
            mask = mask[0:-self.right_padding]

        mask = smooth(mask, window_len=12)
        return mask

    def find_sum_in_section(self):
        assert self.subdocs is not None

        sols = {}
        for i in range(1, 5):
            cap = 'p_cap_solution{}'.format(i)

            solution_section = find_section_by_caption(cap, self.subdocs)
            sols[solution_section] = cap

        results = []
        for solution_section in sols:
            cap = sols[solution_section]
            print(cap)
            # TODO:
            # render_color_text(solution_section.tokens, solution_section.distances_per_pattern_dict[cap])

            x = extract_sum_from_doc(solution_section)
            results.append(x)

        return results
