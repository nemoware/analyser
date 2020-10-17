#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# schemas.py


from analyser.structures import OrgStructuralLevel, ContractSubject, currencly_map

tag_value_field_name = "_value"

charter_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Charter",
  "description": "A charter document attributes",

  "definitions": {

    "tag": {
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

    # "date_tag": {
    #   "allOf": [
    #     {"$ref": "#/definitions/tag"},
    #     {
    #       "properties": {
    #         tag_value_field_name: {
    #           "type": "string",
    #           "format": "date-time"
    #         }
    #       }
    #     }],
    # },

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

          "span": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2
          },

          "span_map": {"type": "string"},
          "confidence": {"type": "number"},

          "competences": {
            "type": "array",
            "items": {"$ref": "#/definitions/competence"},
          }
        }},
      ],

      "required": [tag_value_field_name, "competences", "span"],


    },

    "org": {
      "type": "object",
      "properties": {
        "name": {"$ref": "#/definitions/string_tag"},
        "alias": {"$ref": "#/definitions/string_tag"},
        "type": {"$ref": "#/definitions/string_tag"}
      },
      # "required": ["name", "type"]
    }
  },

  "properties": {
    "date": {
      "$ref": "#/definitions/date_tag"
    },


    #
    # "number": {
    #   "description": "Номер документа",
    #   "$ref": "#/definitions/string_tag"
    # },
    #
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
  },

  "additionalProperties": False
}

# ---------------------------
# self test
# validate(instance={"date":{}}, schema=charter_schema)
