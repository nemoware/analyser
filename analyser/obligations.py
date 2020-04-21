import os

import pandas as pd
from pandas import DataFrame

from analyser.documents import TextMap
from analyser.hyperparams import models_path
from analyser.patterns import CentroidPattensBuilder
from analyser.vocab.obligations_patterns_train_text import obligation_patterns
from analyser.vocab.obligations_patterns_train_text import text as obligations_patterns_train_text


def build_patterns():
  '''
  run when obligations_patterns_train_text is changed
  :return:
  '''
  cp = CentroidPattensBuilder()
  tm = TextMap(obligations_patterns_train_text)
  t_embeddings = cp.tokens_embedder.embedd_tokens(tm.tokens)

  # obligations_patterns
  pat_df = DataFrame(columns=['key'])
  for key, _patterns in obligation_patterns.items():
    date_patterns_emb, _clusterisation_info, embeddings = cp.build_patterns(obligations_patterns_train_text, _patterns,
                                                                            embeddings=t_embeddings)
    _df = DataFrame(date_patterns_emb)
    _df['key'] = key
    # _df['cluster'] = _clusterisation_info['hdbscan']
    pat_df = pd.concat([pat_df, _df], ignore_index=True)
    # _clusterisation_info.to_csv(os.path.join(models_path, key + '_clusterisation_info.csv'))

  pat_df.to_csv(os.path.join(models_path, 'obligations_patterns_emb.csv'))


def read_patterns() -> DataFrame:
  p = os.path.join(models_path, 'obligations_patterns_emb.csv')
  patterns_emb = DataFrame.from_csv(p)
  return patterns_emb


if __name__ == '__main__':
  build_patterns()
  read_patterns()
