#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_parser import ContractDocument3


class ContractAgentsTestCase(unittest.TestCase):

  def test_find_agents(self):
    doc_text = """Акционерное общество «Газпром - Вибраниум и Криптонит» (АО «ГВК»), именуемое в \
    дальнейшем «Благотворитель», в лице заместителя генерального директора по персоналу и \
    организационному развитию Неизвестного И.И., действующего на основании на основании Доверенности № Д-17 от 29.01.2018г, \
    с одной стороны, и Фонд поддержки социальных инициатив «Интерстеларные пущи», именуемый в дальнейшем «Благополучатель», \
    в лице Генерального директора ____________________действующего на основании Устава, с другой стороны, \
    именуемые совместно «Стороны», а по отдельности «Сторона», заключили настоящий Договор о нижеследующем:
    """

    cd = ContractDocument3(doc_text)
    cd.parse()

    _dict ={}
    for tag in cd.agents_tags:
      print(tag)
      _dict[tag.kind]=tag.value

    self.assertEqual('Акционерное Общество', _dict['org.1.type'])
    self.assertEqual('фонд поддержки социальных инициатив', _dict['org.2.type'])
    self.assertEqual('Интерстеларные пущи', _dict['org.2.name'])
    self.assertEqual('Газпром - Вибраниум и Криптонит', _dict['org.1.name'])

    # self.assertEqual('фонд поддержки социальных инициатив ', cd.agents_tags[6]['value'])



if __name__ == '__main__':
  unittest.main()
