#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from typing import List

from enum import Enum, unique


@unique
class OrgStructuralLevel(Enum):
  ShareholdersGeneralMeeting = 0  # Генеральное собрание акционеров
  BoardOfDirectors = 1            # Совет директоров
  CEO = 2                         # Ген дир
  BoardOfCompany = 3              # Правление общества


@unique
class ContractSubject(Enum):
  Transaction = 0  # Сделка
  Сharity = 1      # Благотворительность
  Other = 2        # Другое

class CharterConstraint:
  def __init__(self, upper, lower, subject: ContractSubject, level: OrgStructuralLevel) -> None:
    self.upper, self.lower = upper, lower
    self.subject = subject
    self.level = level


class Charter:
  def __init__(self, org_name, constraints: List[CharterConstraint], date = None) -> None:
    self.org_name = org_name
    self.date = date
    # if not constraints:
    #   self.constraints = {}
    # else:
    #   self.constraints = dict((c.level, c) for c in constraints)

    self.constraints = constraints

  def find_constraint(self, subject: ContractSubject, level):


class Contract:
  def __init__(self, org_name: str, subject: ContractSubject, sum: float, contractor_name: str = None, date = None) -> None:
    self.org_name = org_name
    self.subject = subject
    self.sum = sum
    
    self.contractor_name = contractor_name # FFU
    self.date = date                       # FFU


class Protocol(Contract):
  pass

class FinalViolationLog:

  def __init__(self, charter: Charter) -> None:
    self.charter = charter

  def check_contract(self, contract: Contract):
    self.contract = contract





if __name__ == '__main__':
  l = [
    CharterConstraint(1000, 100, 'subj3', OrgStructuralLevel.BoardOfCompany),
    CharterConstraint(10, 1, 'subj1', OrgStructuralLevel.ShareholdersGeneralMeeting),
    CharterConstraint(100, 10, 'subj2', OrgStructuralLevel.CEO),
  ]

  c = Charter('dd', l)
  print(c.constraints)
