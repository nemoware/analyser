#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from typing import List

from enum import Enum, unique

from ml_tools import TokensWithAttention


@unique
class OrgStructuralLevel(Enum):
  ShareholdersGeneralMeeting = 0  # Генеральное собрание акционеров
  BoardOfDirectors = 1  # Совет директоров
  CEO = 2  # Ген дир
  BoardOfCompany = 3  # Правление общества


@unique
class ContractSubject(Enum):
  Transaction = 0  # Сделка
  Charity = 1  # Благотворительность
  Other = 2  # Другое


class Citation:
  def __init__(self, cite: TokensWithAttention) -> None:
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
    CharterConstraint(1000, 100, ContractSubject.Charity, OrgStructuralLevel.BoardOfCompany),
    CharterConstraint(10, 1, ContractSubject.Other, OrgStructuralLevel.ShareholdersGeneralMeeting),
    CharterConstraint(100, 10, ContractSubject.Transaction, OrgStructuralLevel.CEO),
  ]

  c = Charter('dd', l)
  print(c.constraints)
