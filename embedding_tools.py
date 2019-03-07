from text_tools import *


class AbstractEmbedder:

    # @abstractmethod
    def get_embedding_tensor(self, tokenized_sentences_list):
        pass

    def embedd_tokenized_text(self, words):
        pass


    def embedd_sentence(self, str):
        words = tokenize_text(str)
        return self.embedd_tokenized_text(words)

    def embedd_contextualized_patterns(self, patterns):
        tokenized_sentences_list = []
        regions = {}
        i = 0
        for (ctx_prefix, pattern, ctx_postfix) in patterns:
            sentence = " "
            sentence = sentence.join((ctx_prefix, pattern, ctx_postfix))

            prefix_tokens = tokenize_text(ctx_prefix)
            pattern_tokens = tokenize_text(pattern)
            suffix_tokens = tokenize_text(ctx_postfix)

            start = len(prefix_tokens)
            end = start + len(pattern_tokens)

            sentence_tokens = prefix_tokens+pattern_tokens+suffix_tokens

            print ('embedd_contextualized_patterns', (sentence, start, end))

            regions[i] = (start, end)
            tokenized_sentences_list.append(sentence_tokens)
            i = i + 1

        sentences_emb = self.get_embedding_tensor(tokenized_sentences_list)

        # print(sentences_emb.shape)

        patterns_emb = []

        for i in regions:
            start, end = regions[i]
            # print (start, end)

            sentence_emb = sentences_emb[i]
            pattern_emb = sentence_emb[start:end]

            patterns_emb.append(pattern_emb)

        return np.array(patterns_emb)
