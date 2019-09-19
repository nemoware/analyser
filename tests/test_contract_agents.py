#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_agents import find_org_names_spans, agent_infos_to_tags
from contract_parser import ContractDocument3


class ContractAgentsTestCase(unittest.TestCase):

  def test_find_agents_a(self):
    text = """« Квант Доверия » ( Акционерное общество ) , именуемый в дальнейшем « Какашка » , в лице Вице – Президента - управляющего Филиалом « Квант Доверия » ( Акционерное общество ) в г. Полевском Анонимизированного Анонима Анонимыча , действующего на основании Доверенности с одной стороны , и Общество с ограниченной ответственностью « Газпромнефть-Борьба со Злом » , именуемое в дальнейшем « Принципал » , в лице Генерального директора Иванова Ивана Васильевича , действующего на основании Устава , с другой стороны , именуемые в дальнейшем « Стороны » , заключили настоящее Дополнительное соглашение №3 ( далее по тексту – « Дополнительное соглашение » ) к Договору о выдаче банковских гарантий ( далее по тексту – « Договор » ) о нижеследующем :"""
    # TODO: this sentence is not parceable

    cd = ContractDocument3(text)
    cd.parse()

    agent_infos = find_org_names_spans(cd.tokens_map_norm)
    cd.agents_tags = agent_infos_to_tags(agent_infos)

    _dict = {}
    for tag in cd.agents_tags:
      print(tag)
      _dict[tag.kind] = tag.value

    self.assertIn('org.1.name', _dict)
    self.assertIn('org.2.name', _dict)

    self.assertIn('org.1.alias', _dict)
    self.assertIn('org.2.alias', _dict)

    self.assertIn('org.1.type', _dict)
    self.assertIn('org.2.type', _dict)

    self.assertEqual('Газпромнефть-Борьба со Злом', _dict['org.2.name'])
    self.assertEqual('Квант Доверия', _dict['org.1.name'])
    self.assertEqual('Акционерное Общество', _dict['org.1.type'])
    self.assertEqual('фонд поддержки социальных инициатив', _dict['org.2.type'])

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

    agent_infos = find_org_names_spans(cd.tokens_map_norm)
    cd.agents_tags = agent_infos_to_tags(agent_infos)

    _dict = {}
    for tag in cd.agents_tags:
      print(tag)
      _dict[tag.kind] = tag.value

    self.assertEqual('Акционерное Общество', _dict['org.1.type'])
    self.assertEqual('фонд поддержки социальных инициатив', _dict['org.2.type'])
    self.assertEqual('Интерстеларные пущи', _dict['org.2.name'])
    self.assertEqual('Газпром - Вибраниум и Криптонит', _dict['org.1.name'])

    # self.assertEqual('фонд поддержки социальных инициатив ', cd.agents_tags[6]['value'])


if __name__ == '__main__':
  unittest.main()
