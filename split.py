from patterns import *


def get_sentence_bounds_at_index():
    pass


import numpy.ma as ma


class BasicContractDocument(LegalDocument):

    #   //XXXX:
    def preprocess_text(self, text):
        a = text
        #     a = remove_empty_lines(text)
        a = normalize_text(a, replacements_regex)
        a = self.normalize_sentences_bounds(a)

        return a.lower()

    def split_text_into_sections(self, pattern_factory: AbstractPatternFactory):

        distances_per_section_pattern, __ranges, __winning_patterns = \
            pattern_factory.paragraph_split_pattern.calc_exclusive_distances(self.embeddings, text_right_padding=0)

        # finding pattern positions
        x = distances_per_section_pattern[:, :-TEXT_PADDING]
        indexes_zipped = self.find_sections_indexes(x)

        # self._render_section(indexes_zipped, distances_per_section_pattern, __ranges, __winning_patterns)
        self.section_indexes = indexes_zipped

    def find_sections_indexes(self, distances_per_section_pattern, min_section_size=20):
        x = distances_per_section_pattern
        pattern_to_best_index = np.array([[idx, np.argmin(ma.masked_invalid(row))] for idx, row in enumerate(x)])

        # replace best indices with sentence starts
        pattern_to_best_index[:, 1] = self.find_sentence_beginnings(pattern_to_best_index[:, 1])

        # sort by sentence start
        pattern_to_best_index = np.sort(pattern_to_best_index.view('i8,i8'), order=['f1'], axis=0).view(np.int)

        # remove "duplicated" indexes
        return self.remove_similar_indexes(pattern_to_best_index, 1, min_section_size)

    def remove_similar_indexes(self, indexes, column, min_section_size=20):
        indexes_zipped = []
        indexes_zipped.append(indexes[0])

        for i in range(1, len(indexes)):
            if indexes[i][column] - indexes[i - 1][column] > min_section_size:
                pattern_to_token = indexes[i]
                indexes_zipped.append(pattern_to_token)
        return np.squeeze(indexes_zipped)

    def find_sentence_beginnings(self, best_indexes):
        return [find_token_before_index(self.tokens, i, '\n', 0) for i in best_indexes]

    def get_subject_ranges(self, indexes_zipped):
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

        self.split_text_into_sections(pattern_fctry)
        indexes_zipped = self.section_indexes

        head_range, subj_range = self.get_subject_ranges(indexes_zipped)

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

    def _find_sum(self, pattern_factory):

        assert TEXT_PADDING > 0

        min_i, sums_no_padding, confidence = pattern_factory.sum_pattern.find(self.embeddings, TEXT_PADDING)

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

        currency_re = re.compile(r'((^|\s+)(\d+[., ])*\d+)(\s*([(].{0,100}[)]\s*)?(евро|руб))')

        sentence = untokenize(sentence_tokens).lower().strip()

        r = currency_re.findall(sentence)

        f = None
        try:
            result = r[0][5]
            f = (float(r[0][0].replace(" ", "").replace(",", ".")), r[0][5])
        except:
            pass

        self.found_sum = (f, (start, end), sentence, meta)

    #     return

    def analyze(self, pattern_factory):
        self.embedd(pattern_factory)
        self._find_sum(pattern_factory)

        self.subj_ranges, self.winning_subj_patterns = self.find_subject_section(
            pattern_factory, {"charity": [0, 5], "commerce": [5, 5 + 7]})
