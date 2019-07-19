#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from charter_parser import CharterDocumentParser
from charter_patterns import CharterPatternFactory
from contract_parser import ContractAnlysingContext, ContractDocument3

from doc_structure import remove_similar_indexes_considering_weights
from embedding_tools import AbstractEmbedder
from legal_docs import *
from parsing import print_prof_data
from patterns import *




class ContractAgentsTestCase(unittest.TestCase):

  def test_find_agents(self):

    doc_text="""Акционерное общество «Газпромнефть – Московский НПЗ» (АО «Газпромнефть-МНПЗ»), именуемое в \
    дальнейшем «Благотворитель», в лице заместителя генерального директора по персоналу и \
    организационному развитию Зыкова Д.В., действующего на основании на основании Доверенности № Д-17 от 29.01.2018г, \
    с одной стороны, и Фонд поддержки социальных инициатив «Родные города», именуемый в дальнейшем «Благополучатель», \
    в лице Генерального директора ____________________действующего на основании Устава, с другой стороны, \
    именуемые совместно «Стороны», а по отдельности «Сторона», заключили настоящий Договор о нижеследующем:
    """

    cd =ContractDocument3(doc_text)
    cd.parse()


    print(cd.agents_tags)


if __name__ == '__main__':
  unittest.main()
