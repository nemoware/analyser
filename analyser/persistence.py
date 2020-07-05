import logging

import numpy as np

from analyser.documents import TextMap, CaseNormalizer
from analyser.legal_docs import Paragraph
from analyser.ml_tools import SemanticTag
from integration.word_document_parser import join_paragraphs


class DbJsonDoc:

  def __init__(self, j: dict):
    self.analysis = None
    self.state: int = -1
    self.parse = None
    self.filename = None
    self.user = None
    self._id = None
    self.retry_number: int = 0
    self.__dict__.update(j)

  def asLegalDoc(self):
    doc = join_paragraphs(self.parse, self._id, filename=self.filename)
    if self.analysis is not None and ('normal_text' in self.analysis):

      doc.tokens_map_norm = self.get_tokens_for_embedding()
      doc.tokens_map = self.get_tokens_map_unchaged()

      if len(doc.tokens_map_norm) != len(doc.tokens_map):
        msg = f'{doc._id} has wrong tokenization: len(doc.tokens_map_norm)={len(doc.tokens_map_norm)}; len( doc.tokens_map)={len(doc.tokens_map)} '
        logging.error(msg)

      headers = self.analysis.get('headers', None)
      if headers is not None:
        doc.paragraphs = []
        last = len(doc.tokens_map)
        for i, h in enumerate(headers):
          header_tag = SemanticTag('headline', h['value'], h['span'])
          body_end = last
          if i < len(headers) - 1:
            body_end = headers[i + 1]['span'][0]
          bodyspan = header_tag.span[1] + 1, body_end
          body_tag = SemanticTag('paragraphBody', None, bodyspan)

          para = Paragraph(header_tag, body_tag)
          doc.paragraphs.append(para)

    return doc

  def get_tokens_map_unchaged(self):
    _map = self.analysis['tokenization_maps']['words']
    tokens_map = TextMap(self.analysis['normal_text'], _map)
    return tokens_map

  def get_tokens_for_embedding(self):
    _tokens_map = self.get_tokens_map_unchaged()
    tokens_map_norm = CaseNormalizer().normalize_tokens_map_case(_tokens_map)
    return tokens_map_norm

  def as_dict(self):
    return self.__dict__

  def __len__(self):
    arrr = self.analysis['tokenization_maps']['words']
    return len(arrr)

  def get_attributes(self) -> dict:
    if self.user is not None:
      attributes = self.user['attributes']
    else:
      attributes = self.analysis['attributes']
    return attributes

  def get_subject(d):
    return d.get_attribute('subject')

  def get_attribute_value(d, attr):
    a = d.get_attribute(attr)
    if a is not None:
      return a['value']
    return None

  def get_attribute(d, attr):
    atts = d.get_attributes()
    if attr in atts:
      return atts[attr]
    else:
      return {
        'value': None,
        'confidence': 0.0,
        'span': [np.nan, np.nan]
      }  ## fallback for safety

  def get_attr_span_start(self, a):
    att = self.get_attributes()
    if a in att:
      return att[a]['span'][0]
