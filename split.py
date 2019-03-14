from patterns import *


def get_sentence_bounds_at_index():
    pass


import numpy.ma as ma


class ContractDocument(LegalDocument):

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

        self._render_section(indexes_zipped, distances_per_section_pattern, __ranges, __winning_patterns)
        self.section_indexes = indexes_zipped

    def find_sections_indexes(self, distances_per_section_pattern, min_section_size=20):
        x = distances_per_section_pattern
        pattern_to_best_index = np.array([[idx, np.argmin(ma.masked_invalid(row))] for idx, row in enumerate(x)])
        print("pattern_to_best_index\n", pattern_to_best_index, len(pattern_to_best_index))

        # replace best indices with sentence starts
        pattern_to_best_index[:, 1] = self.find_sentence_beginnings(pattern_to_best_index[:, 1])

        print("pattern_to_best_index\n", pattern_to_best_index, len(pattern_to_best_index))

        # sort by sentence start
        pattern_to_best_index = np.sort(pattern_to_best_index.view('i8,i8'), order=['f1'], axis=0).view(np.int)

        print("pattern_to_best_index sorted\n", pattern_to_best_index, len(pattern_to_best_index))
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

    def _find_sentence_beginnings(self, best_indexes):

        sentence_starts = {}
        for i in best_indexes:
            start = find_token_before_index(self.tokens, i[1], '\n')
            if start == -1: start = 0
            sentence_starts[i[0]] = start

        sentence_starts = [[st, sentence_starts[st]] for st in sorted(sentence_starts.keys())]
        return sentence_starts

    def find_subject_section(self, PF: AbstractPatternFactory, number_of_charity_patterns):

        self.split_text_into_sections(PF)
        indexes_zipped = self.section_indexes

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

        distances_per_subj_pattern, ranges_, winning_patterns = PF.subject_patterns.calc_exclusive_distances(
            self.embeddings,
            text_right_padding=0)
        distances_per_pattern_t = distances_per_subj_pattern[:, subj_range.start:subj_range.stop]

        ranges = []
        for row in distances_per_pattern_t:
            b = np.array(list(filter(lambda x: not np.isnan(x), row)))
            if len(b):
                min = b.min()
                max = b.max()
                mean = b.mean()
                ranges.append([min, max, mean])
            else:
                _id = len(ranges)
                print("WARNING: never winning pattern detected! index:", _id)
                ranges.append([np.inf, -np.inf, 0])

        weight_per_pat = np.zeros((13, 3))
        for token_key in winning_patterns:
            (pattern_id, weight) = winning_patterns[token_key]

            weight_per_pat[pattern_id][0] += weight
            weight_per_pat[pattern_id][1] += 1

        for l in weight_per_pat:
            _mean = l[1] / l[0]
            l[2] = _mean

        chariy_slice = weight_per_pat[0:5, 2:3]
        commerce_slice = weight_per_pat[6:6 + 7, 2:3]
        min_charity_index = min_index(chariy_slice)
        min_commerce_index = min_index(commerce_slice)

        self.per_subject_distances = [chariy_slice[min_charity_index][0], chariy_slice[min_commerce_index][0]]

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
            pattern_factory, number_of_charity_patterns=5)
