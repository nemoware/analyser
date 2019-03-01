from text_tools import *


class AbstractEmbedder:

    # @abstractmethod
    def get_embedding_tensor(self, str):
        pass

    def embedd_tokenized_text(self, words):
        pass


    def embedd_sentence(self, str):
        words = tokenize_text(str)
        return self.embedd_tokenized_text(words)

    def embedd_contextualized_patterns(self, patterns):
        sentences = []
        regions = {}
        i = 0
        for (ctx_prefix, pattern, ctx_postfix) in patterns:
            sentence = " "
            sentence = sentence.join((ctx_prefix, pattern, ctx_postfix))

            start = len(tokenize_text(ctx_prefix))
            end = start + len(tokenize_text(pattern))

            print ((sentence, start, end))

            regions[i] = (start, end)
            sentences.append(sentence)
            i = i + 1

        sentences_emb = self.get_embedding_tensor(sentences)

        # print(sentences_emb.shape)

        patterns_emb = []

        for i in regions:
            start, end = regions[i]
            # print (start, end)

            sentence_emb = sentences_emb[i]
            pattern_emb = sentence_emb[start:end]

            patterns_emb.append(pattern_emb)

        return patterns_emb
