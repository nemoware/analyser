#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from typing import List

from enum import Enum, unique, EnumMeta

from ml_tools import TokensWithAttention


class DisplayStringEnumMeta(EnumMeta):
  def __new__(mcs, name, bases, attrs):
    obj = super().__new__(mcs, name, bases, attrs)
    obj._value2member_map_ = {}
    for m in obj:
      value, display_string = m.value
      m._value_ = value
      m.display_string = display_string
      obj._value2member_map_[value] = m

    return obj


@unique
class OrgStructuralLevel(Enum, metaclass=DisplayStringEnumMeta):
  ShareholdersGeneralMeeting = 0, 'Генеральное собрание акционеров'
  BoardOfDirectors = 1,           'Совет директоров'
  CEO = 2,                        'Генеральный директор'
  BoardOfCompany = 3,             'Правление общества'

@unique
class ContractSubject(Enum, metaclass=DisplayStringEnumMeta):
  Deal = 0,       'Сделка'
  Charity = 1,    'Благотворительность'
  Other = 2,      'Другое'
  Lawsuit = 3,    'Судебные издержки'
  RealEstate = 4, 'Недвижимость'


class Citation:
  def __init__(self, cite: TokensWithAttention = None) -> None:
    self.citation = cite


class CharterConstraint(Citation):
  def __init__(self, upper, lower, subject: ContractSubject, level: OrgStructuralLevel, *args, **kwargs) -> None:
    super(CharterConstraint, self).__init__(*args, **kwargs)
    self.upper, self.lower = upper, lower
    self.subject = subject
    self.level = level


class Charter(Citation):
  def __init__(self, org_name, constraints: List[CharterConstraint], date=None, *args, **kwargs) -> None:
    super(Charter, self).__init__( *args, **kwargs)
    self.org_name = org_name
    self.date = date
    # if not constraints:
    #   self.constraints = {}
    # else:
    #   self.constraints = dict((c.level, c) for c in constraints)

    self.constraints = constraints

  def find_constraint(self, subject: ContractSubject):
    pass


class Contract(Citation):
  def __init__(self, org_name: str, subject: ContractSubject, sum: float, contractor_name: str = None,
               date=None, *args, **kwargs) -> None:
    super(Contract, self).__init__( *args, **kwargs)
    self.org_name = org_name
    self.subject = subject
    self.sum = sum

    self.contractor_name = contractor_name  # FFU
    self.date = date  # FFU


class Protocol(Contract):
  def __init__(self, *args, **kwargs) -> None:
    super(Protocol, self).__init__( *args, **kwargs)


class FinalViolationLog:

  def __init__(self, charter: Charter) -> None:
    self.charter = charter

  def check_contract(self, contract: Contract):
    self.contract = contract
    pass


if __name__ == '__main__':
  l = [
    CharterConstraint(1000, 100, ContractSubject.Charity, OrgStructuralLevel.BoardOfCompany, cite = None),
    CharterConstraint(10, 1, ContractSubject.Other, OrgStructuralLevel.ShareholdersGeneralMeeting, cite = None),
    CharterConstraint(100, 10, ContractSubject.Deal, OrgStructuralLevel.CEO, cite = None),
  ]

  c = Charter('dd', l, cite = None)
  print(c.constraints)

  print(f'{ContractSubject.Charity.name}')
  print(f'{ContractSubject.Charity.display_string}')
  print('All:')
  for subj in ContractSubject:
    print(f'{subj.display_string}')

  for level in OrgStructuralLevel:
    print(f'{level.display_string}')
