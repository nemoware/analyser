import hdbscan
import pandas as pd
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity

from analyser.documents import TextMap
from analyser.hyperparams import HyperParameters
from analyser.ml_tools import max_exclusive_pattern, get_centroids, \
  Embeddings
from analyser.text_tools import *
from tf_support.embedder_elmo import ElmoEmbedder


class CentroidPattensBuilder:
  def __init__(self):
    self.sentence_embedder = ElmoEmbedder.get_instance('default')
    self.tokens_embedder = ElmoEmbedder.get_instance('elmo')

  def build_patterns(self, text, pattens, embeddings=None):
    tm = TextMap(text)

    if embeddings is None:
      embeddings = self.tokens_embedder.embedd_tokens(tm.tokens)
    patterns_emb = self.sentence_embedder.embedd_tokens(pattens)

    print('embeddings.shape', embeddings.shape)
    print('patterns_emb.shape', patterns_emb.shape)

    distances = np.array(cosine_similarity(embeddings, patterns_emb)).T

    attention_vector = max_exclusive_pattern(distances)

    df: DataFrame = pd.DataFrame()
    df['dist'] = attention_vector
    df['tokens'] = tm.tokens

    selected_indices = [i for i, v in enumerate(attention_vector) if
                        v > HyperParameters.obligations_date_pattern_threshold and len(tm.tokens[i].strip()) > 2]
    selected_embeddings = embeddings[selected_indices]
    selection: DataFrame = df.loc[selected_indices]

    # clusterize
    cluster_label_col_name = self.clusterize(selected_embeddings, selection)

    cluster_names = selection[cluster_label_col_name].unique()  # np.unique(clusterer.labels_)
    nclasses = len(cluster_names)
    print('number_of_classes', nclasses)

    _well_clustered = selection[
      selection[cluster_label_col_name + '_probabilities'] > HyperParameters.hdbscan_cluster_proximity]
    centroids = get_centroids(embeddings, _well_clustered, 'hdbscan')

    return centroids, _well_clustered, embeddings

  @staticmethod
  def clusterize(selected_embeddings: Embeddings, resulting_df: DataFrame, min_cluster_size=5):

    if len(selected_embeddings) != len(resulting_df):
      raise AssertionError('len(selected_embeddings) != len(resulting_df)')

    print('clustering with HDBSCAN....')
    clusterer = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=min_cluster_size)
    clusterer.fit(selected_embeddings.astype(np.double))

    resulting_df['hdbscan'] = clusterer.labels_
    resulting_df['hdbscan_probabilities'] = clusterer.probabilities_

    return 'hdbscan'

  def calc_patterns_centroids(self, patterns_dict):
    '''
    1. embedd patterns as sentences
    2. clusterize embeddings
    3. caclulate cluster centers
    :param patterns_dict:
    :return:
    '''
    for k in patterns_dict:
      arr = patterns_dict[k]
      patterns_emb = self.sentence_embedder.embedd_tokens(arr)
      print(f'{k} patterns_emb.shape', patterns_emb.shape)

      # clusterize
      df = DataFrame()
      cluster_label_col_name = self.clusterize(patterns_emb, df)

      cluster_names = df[cluster_label_col_name].unique()  # np.unique(clusterer.labels_)
      nclasses = len(cluster_names)
      print(f'{k} number_of_classes', nclasses)
