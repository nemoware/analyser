import nltk
import scipy.spatial.distance as distance
import numpy as np
import string

nltk.download('punkt')


def replace_with_map(txt, replacements):
    a = txt
    for (src, target) in replacements:
        a = a.replace(src, target)

    return a


def remove_empty_lines(original_text):
    a = "\n".join([ll.strip() for ll in original_text.splitlines() if ll.strip()])
    return a.replace('\t', ' ')


def tokenize_text(text):
    sentences = text.split('\n')
    result = []
    for sentence in sentences:
        result += nltk.word_tokenize(sentence)
        result += ['\n']

    return result


# ----------------------------------------------------------------
# DISTANCES
# ----------------------------------------------------------------

def dist_hausdorff(u, v):
    return max(distance.directed_hausdorff(v, u, 4)[0], distance.directed_hausdorff(u, v, 4)[0]) / 40


def dist_mean_cosine(u, v):
    return distance.cosine(u.mean(0), v.mean(0))


def dist_mean_eucl(u, v):
    return distance.euclidean(u.mean(0), v.mean(0))


def dist_sum_cosine(u, v):
    return distance.cosine(u.sum(0), v.sum(0))


# not in use
def dist_correlation_min_mean(u, v):
    if u.shape[0] > v.shape[0]:
        return distance.cdist(u, v, 'correlation').min(0).mean()
    else:
        return distance.cdist(v, u, 'correlation').min(0).mean()


def dist_cosine_min_mean(u, v):
    if u.shape[0] > v.shape[0]:
        return distance.cdist(u, v, 'cosine').min(0).mean()
    else:
        return distance.cdist(v, u, 'cosine').min(0).mean()


# not in use
def dist_euclidean_min_mean(u, v):
    if u.shape[0] > v.shape[0]:
        return distance.cdist(u, v, 'euclidean').min(0).mean()
    else:
        return distance.cdist(v, u, 'euclidean').min(0).mean()


# ----------------------------------------------------------------
# MISC
# ----------------------------------------------------------------


def norm_matrix(mtx):
    mtx = mtx - mtx.min()
    mtx = mtx / np.abs(mtx).max()
    return mtx


def min_index(sums):
    min_i = 0
    min_d = float('Infinity')

    for d in range(0, len(sums)):
        if sums[d] < min_d:
            min_d = sums[d]
            min_i = d

    return min_i


# ----------------------------------------------------------------
# TOKENS
# ----------------------------------------------------------------

def sentence_similarity_matrix(emb, distance_function):
    mtx = np.zeros(shape=(emb.shape[0], emb.shape[0]))

    # TODO: write it Pythonish!!
    for u in range(emb.shape[0]):
        for v in range(emb.shape[0]):
            mtx[u, v] = distance_function(emb[u], emb[v])

    # TODO: no norm here
    return mtx  # norm_matrix(mtx)


def untokenize(tokens):
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
