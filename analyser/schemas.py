#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# schemas.py
from analyser.ml_tools import SemanticTagBase
from analyser.structures import OrgStructuralLevel, ContractSubject, currencly_map

tag_value_field_name = "value"


class DocumentSchema:
  date: SemanticTagBase
  number: SemanticTagBase

  def __init__(self):
    super().__init__()


class HasOrgs(SemanticTagBase):
  def __init__(self):
    super().__init__()
    self.orgs: [OrgItem] = []


class ContractPrice(SemanticTagBase):
  def __init__(self):
    super().__init__()

    self.amount: SemanticTagBase  # netto or brutto #deprecated
    self.currency: SemanticTagBase
    self.sign: SemanticTagBase
    self.vat: SemanticTagBase  # number
    self.vat_unit: SemanticTagBase  # percentage
    self.value_brutto: SemanticTagBase  # netto + VAT
    self.value_netto: SemanticTagBase  # value before VAT


class AgendaItemContract(HasOrgs):
  number: SemanticTagBase
  date: SemanticTagBase
  price: ContractPrice = None

  def __init__(self):
    super().__init__()


class AgendaItem(SemanticTagBase):
  solution: SemanticTagBase or None = None

  def __init__(self):
    super().__init__()
    self.contract: AgendaItemContract or None = None


class OrgItem():
  def __init__(self):
    super().__init__()
    self.type: SemanticTagBase
    self.name: SemanticTagBase


class ContractSchema(DocumentSchema, HasOrgs):
  price: ContractPrice = None

  def __init__(self):
    super().__init__()
    # self.orgs: [OrgItem] = []
    self.date: SemanticTagBase
    self.number: SemanticTagBase
    self.subject: SemanticTagBase


class ProtocolSchema(DocumentSchema):

  def __init__(self):
    super().__init__()
    self.org: OrgItem = OrgItem()
    self.structural_level: SemanticTagBase
    self.agenda_items: [AgendaItem] = []


class CharterConstraint:
  def __init__(self):
    super().__init__()
    self.margins: [ContractPrice] = []


class Competence(SemanticTagBase):
  def __init__(self):
    super().__init__()
    # self.value: OrgStructuralLevel = None
    self.constraints: [CharterConstraint] = []


class CharterStructuralLevel(SemanticTagBase):
  def __init__(self):
    super().__init__()
    # self.value: OrgStructuralLevel = None
    self.competences: [Competence] = []


class CharterSchema(DocumentSchema):
  org: OrgItem = OrgItem()

  def __init__(self):
    super().__init__()
    self.structural_levels: [CharterStructuralLevel] = []


document_schemas = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "Legal document attributes",
  "description": "Legal document attributes. Schema draft 4 is used for compatibility with Mongo DB",

  "definitions": {

    "tag": {
      "description": "a piece of text, denoting an attributes",
      "type": "object",

      "properties": {
        "span": {
          "type": "array",
          "minItems": 2,
          "maxItems": 2
        },
        "confidence": {
          "type": "number"
        },
        "span_map": {
          "type": "string"
        }
      },
      "required": ["span", tag_value_field_name]
    },

    "string_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "string"
            }
          },

        }]
    },

    "boolean_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "boolean"
            }
          },

        }]

    },

    "number_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "number"
            }
          }
        }],
    },

    "date_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "string",
              "format": "date-time"
            }
          },

        }],

    },

    "agenda_contract": {
      "description": "Атрибуты контракта, о котором идет речь в повестке",

      "properties": {

        "number": {
          "$ref": "#/definitions/string_tag"
        },

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "solution": {
          "$ref": "#/definitions/boolean_tag"
        },

        "warnings": {
          "type": "string"
        },

        "orgs": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/contract_agent",
          }
        },

        "value": {
          "$ref": "#/definitions/currency_value"
        },

      },
      "additionalProperties": False

    },
    "agenda": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            "contract": {"$ref": "#/definitions/agenda_contract"}
          },
          "required": ["span"],
          # "additionalProperties": False
        }],
    },

    "currency": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "enum": list(currencly_map.values())
            }
          }
        }],
    },

    "sign": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "integer",
              "enum": [-1, 0, 1]
            }
          }}]
    },

    "currency_value": {

      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            "value": {
              "$ref": "#/definitions/number_tag",
            },

            "currency": {
              "$ref": "#/definitions/currency"
            },

            "sign": {
              "$ref": "#/definitions/sign",
            }

          },
          "required": ["sign", "value", "currency"],
        }
      ],

    },

    "competence": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "enum": ContractSubject._member_names_
            },
            "constraints": {
              "type": "array",
              "items": {"$ref": "#/definitions/currency_value"}
            }
          }}],
    },

    "structural_level": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {"properties": {
          tag_value_field_name: {
            "enum": OrgStructuralLevel._member_names_
          },

          "competences": {
            "type": "array",
            "items": {"$ref": "#/definitions/competence"},
          }
        }},
      ],

    },

    "org": {
      "type": "object",
      "properties": {
        "name": {"$ref": "#/definitions/string_tag"},
        "type": {"$ref": "#/definitions/string_tag"}
      },
      # "required": ["name", "type"]
    },

    "contract_agent": {
      "allOf": [
        {"$ref": "#/definitions/org"},
        {
          "properties": {
            "alias": {"$ref": "#/definitions/string_tag"},
          }
        }
      ]

      # "required": ["name", "type"]
    }
  },

  "properties": {

    "charter": {
      "properties": {

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "orgs": {
          "type": "array",
          "maxItems": 1,
          "uniqueItems": True,
          "items": {
            "$ref": "#/definitions/org",
          }
        },

        "structural_levels": {
          "type": "array",
          "uniqueItems": True,
          "items": {
            "$ref": "#/definitions/structural_level"
          }
        }
      }
    },

    "contract": {
      "properties": {

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "number": {
          "$ref": "#/definitions/number_tag"
        },

        "orgs": {
          "type": "array",
          "maxItems": 10,
          "uniqueItems": True,
          "items": {
            "$ref": "#/definitions/org",
          }
        },

      }
    },

    "protocol": {
      "properties": {

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "number": {
          "$ref": "#/definitions/number_tag"
        },

        "structural_level": {
          "$ref": "#/definitions/structural_level"
        },

        "orgs": {
          "type": "array",
          "maxItems": 1,
          "items": {
            "$ref": "#/definitions/org",
          }
        },

        "agenda_items": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/agenda",
          }
        },

      },
      "additionalProperties": False
    },

  },

}

# ---------------------------
# self test
# validate(instance={"date":{}}, schema=charter_schema)
