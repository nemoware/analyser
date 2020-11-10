#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from jsonschema import validate, ValidationError, FormatChecker

from analyser.schemas import document_schemas


class TestSchema(unittest.TestCase):
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
