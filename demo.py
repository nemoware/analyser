#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from contract_parser import default_contract_parsing_config
from legal_docs import LegalDocument
from legal_docs import tokenize_text
from ml_tools import *

# subject_types = {
#   'charity': 'благотворительность'.upper(),
#   'comm': 'коммерческая сделка'.upper(),
#   'comm_estate': 'недвижемость'.upper(),
#   'comm_service': 'оказание услуг'.upper()
# }
#
# subject_types_dict = {**subject_types, **{'unknown': 'предмет догоовора не ясен'}}

default_contract_parsing_config.headline_attention_threshold = 0.9





class ContractDocument2(LegalDocument):
  def __init__(self, original_text: str):
    LegalDocument.__init__(self, original_text)
    self.subject = ('unknown', 1.0)
    self.contract_values = [ProbableValue]

  def tokenize(self, _txt):
    return tokenize_text(_txt)

##---------------------------------------##---------------------------------------##---------------------------------------



