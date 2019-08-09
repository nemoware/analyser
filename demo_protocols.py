# -*- coding: utf-8 -*-
from patterns import *
from renderer import AbstractRenderer


class ProtocolRenderer(AbstractRenderer):
  pass


# from split import *
# ----------------------


"""### Define Patterns"""

"""## Processing"""

from collections import Counter


def subject_weight_per_section(doc, subj_pattern, paragraph_split_pattern):
  assert doc.section_indexes is not None

  distances_per_subj_pattern_, ranges_, winning_patterns = subj_pattern.calc_exclusive_distances(
    doc.embeddings)

  ranges_global = [
    np.nanmin(distances_per_subj_pattern_),
    np.nanmax(distances_per_subj_pattern_)]

  section_names = [[paragraph_split_pattern.patterns[s[0]].name, s[1]] for s in doc.section_indexes]
  voting: List[str] = []
  for i in range(1, len(section_names)):
    p1 = section_names[i - 1]
    p2 = section_names[i]

    distances_per_pattern_t = distances_per_subj_pattern_[:, p1[1]:p2[1]]

    dist_per_pat = []
    for row in distances_per_pattern_t:
      dist_per_pat.append(np.nanmin(row))

    patindex = np.nanargmin(dist_per_pat)
    pat_prefix = subj_pattern.patterns[patindex].name[:5]
    #         print(patindex, pat_prefix)

    voting.append(pat_prefix)

    ## HACK more attention to particular sections
    if p1[0] == 'p_agenda' or p1[0] == 'p_solution' or p1[0] == 'p_question':
      voting.append(pat_prefix)

  return Counter(voting), ranges_global, winning_patterns
