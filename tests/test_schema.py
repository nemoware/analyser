#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from jsonschema import validate, ValidationError, FormatChecker

from analyser.ml_tools import SemanticTagBase
from analyser.schemas import CharterSchema, CharterStructuralLevel, Competence, ContractPrice, OrgItem, \
  Schema2LegacyListConverter
from analyser.schemas import document_schemas
from analyser.structures import OrgStructuralLevel, ContractSubject


class TestSchema(unittest.TestCase):

  def test_convert_to_legasy_list(self):
    cs = CharterSchema()

    cs.org = OrgItem()
    cs.org.name = SemanticTagBase()
    cs.org.type = SemanticTagBase()
    cs.org.alias = SemanticTagBase()

    cs.date = SemanticTagBase()

    structural_level = CharterStructuralLevel()
    structural_level.value = OrgStructuralLevel.BoardOfCompany
    structural_level.span = (1, 2)

    structural_level1 = CharterStructuralLevel()
    structural_level1.value = OrgStructuralLevel.AllMembers
    structural_level1.span = (1, 2)

    structural_level.confidence = 0.777
    cs.structural_levels.append(structural_level)
    cs.structural_levels.append(structural_level1)

    comp = Competence()
    comp.confidence = 0.22
    comp.span = (2, 2)
    comp.value = ContractSubject.Charity
    structural_level.competences.append(comp)

    cp1 = ContractPrice()
    cp1.sign = SemanticTagBase()
    cp1.sign.value = -1

    cp2 = ContractPrice()
    cp2.sign = SemanticTagBase()
    cp2.sign.value = 1
    cp2.amount = SemanticTagBase()
    cp2.amount.value = 1
    cp2.currency = SemanticTagBase()
    cp2.currency.value = "USD"

    cp3 = ContractPrice()
    cp3.amount = SemanticTagBase()

    comp.constraints = []
    comp.constraints.append(cp1)
    comp.constraints.append(cp2)
    comp.constraints.append(cp3)

    converter = Schema2LegacyListConverter()
    dest = {}
    converter.schema2list(dest, cs)
    for k, v in dest.items():
      print(f"[{k}]", v)

    self.assertTrue('BoardOfCompany/Charity/constraint-min-1' in dest)
    self.assertTrue('org-1-name' in dest)
    self.assertTrue('org-1-type' in dest)
    self.assertTrue('org-1-alias' in dest)
    # print(dest)

  def test_date_wrong_2(self):
    tree = {
      "contract": {
        "date": {
          # "_ovalue": "2017-06-13T00:00:00.000Z",
          "value": "wrong date",
          "span": [14, 17],
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    with self.assertRaises(ValidationError) as context:
      validate(instance=tree, schema=document_schemas, format_checker=FormatChecker())

    self.assertIsNotNone(context.exception)
    print(context.exception)

  def test_date_wrong_3(self):
    tree = {
      "contract": {
        "date": {
          "_ovalue": "2017-06-13T00:00:00.000Z",
          "span": [14, 17],
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    with self.assertRaises(ValidationError) as context:
      validate(instance=tree, schema=document_schemas, format_checker=FormatChecker())

    self.assertIsNotNone(context.exception)
    print(context.exception)

  def test_date_wrong(self):
    tree = {
      "contract": {
        "date": {
          "value": "2017-06-13T00:00:00.000Z",
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    with self.assertRaises(ValidationError) as context:
      validate(instance=tree, schema=document_schemas)

    self.assertIsNotNone(context.exception)

  def test_date_correct(self):
    tree = {
      "contract": {
        "date": {
          "value": "2017-06-13T00:00:00.000Z",
          "span": [14, 17],
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    validate(instance=tree, schema=document_schemas)

  def test_org_correct(self):
    tree = {
      "contract": {"orgs": [{
        "name": {
          "value": "ГПН",
          "span": [30, 31],
          "span_map": "words",
          "confidence": 0.8
        },
        "type": {
          "value": "Акционерное общество",
          "span": [27, 29],
          "span_map": "words",
          "confidence": 0.8
        }
      }]}
    }

    validate(instance=tree, schema=document_schemas)

  def test_org_wrong(self):
    tree = {
      "contract": {
        "orgs": [{
          "name": {
            "Xvalue": "ГПН",
            "span": [30, 31],
            "span_map": "words",
            "confidence": 0.8
          },
          "type": {
            "Xvalue": "Акционерное общество",
            "span": [27, 29],
            "span_map": "words",
            "confidence": 0.8
          }
        }]}
    }

    with self.assertRaises(ValidationError) as context:
      validate(instance=tree, schema=document_schemas)

    self.assertIsNotNone(context.exception)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
