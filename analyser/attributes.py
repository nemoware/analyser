from bson.objectid import ObjectId
from jsonschema import validate

from analyser.finalizer import get_doc_by_id
from analyser.structures import OrgStructuralLevel, ContractSubject
from integration.db import get_mongodb_connection


def convert_org(attr_name: str, path: str, v, tree):
  orgs_arr = tree.get('orgs', [])
  tree['orgs'] = orgs_arr

  name_parts = attr_name.split('-')
  _index = int(name_parts[1]) - 1
  pad_array(orgs_arr, _index + 1)

  org_node = orgs_arr[_index]

  _v = {}
  copy_attr(v, _v)
  org_node[name_parts[-1]] = _v
  # orgs_arr[_index] = v
  pass


def copy_attr(src, dest):
  for key in ['span', 'span_map', 'confidence']:
    dest[key] = src.get(key)
  dest['_value'] = src.get('value')


def getput_node(subtree, key_name, defaultValue):
  _node = subtree.get(key_name, defaultValue)
  subtree[key_name] = _node  # put_it
  return _node


def pad_array(arr, size):
  for _i in range(len(arr), size):
    arr.append({})


def get_or_create_constraint_node(path, tree):
  path_s: [] = path.split('/')

  # attr_name: str = path_s[-1]

  _structural_levels_dict = getput_node(tree, 'structural_levels', {})
  
  structural_level_node = getput_node(_structural_levels_dict, path_s[0], {})
  if len(path_s) == 1:
    return structural_level_node

  subject_node = getput_node(structural_level_node, path_s[1], {})
  if len(path_s) == 2:
    return subject_node

  constraint = path_s[2].split('-')
  constraint_margin = constraint[1]  # min max
  constraint_margin_index = 0
  if len(constraint) == 3:
    constraint_margin_index = int(constraint[2]) - 1

  margin_nodes = getput_node(subject_node, constraint_margin, [])
  pad_array(margin_nodes, constraint_margin_index + 1)
  margin_node = margin_nodes[constraint_margin_index]
  if len(path_s) == 3:
    return margin_node

  summ_node = getput_node(margin_node, path_s[3], {})
  return summ_node


def convert_constraints(attr_name: str, path: str, v, tree):
  node = get_or_create_constraint_node(path, tree)
  if node is not None:
    copy_attr(v, node)


def remove_empty_from_list(lst):
  l = [v for v in lst if len(v) > 0]
  return l


def clean_up_tree(tree):
  _structural_levels_dict: {} = tree.get('structural_levels', {})
  for root_s in OrgStructuralLevel._member_names_:
    print(root_s)
    s_node = _structural_levels_dict.get(root_s, {})

    for subj_s in ContractSubject._member_names_:
      subj_node = s_node.get(subj_s, {})
      print(subj_s)

      mins_list = remove_empty_from_list(subj_node.get("min", []))
      maxs_list = remove_empty_from_list(subj_node.get("max", []))

      subj_node['min'] = mins_list
      subj_node['max'] = maxs_list


def convert_charter_db_attributes_to_tree(attrs):
  _index = {}
  tree = {}

  for path, v in attrs.items():
    key_s: [] = path.split('/')
    attr_name: str = key_s[-1]
    root_s = key_s[0]
    print(attr_name, '\t\t', path, v)

    if v.get('span', None):
      #   it's what we need
      print(v)

      # handle org
      if attr_name.startswith('org-'):
        convert_org(attr_name, path, v, tree)

      # handle date
      if (root_s == 'date'):
        _v = {}
        copy_attr(v, _v)
        tree[attr_name] = _v



      if (root_s in OrgStructuralLevel._member_names_):
        convert_constraints(attr_name, path, v, tree)

    #
    # if (len(key_s) == 4 and root_s in OrgStructuralLevel._member_names_):
    #
    #   structural_level_constraints = structural_levels_dict.get(root_s, {})
    #   structural_levels_dict[root_s] = structural_level_constraints
    #
    #   constraint_subj = key_s[1]
    #   constraint_edge = key_s[2].split('-')[1]
    #
    #   constraint_v = key_s[3]
    #
    #
    #   constraint_subj_node = structural_level_constraints.get(constraint_subj, {})
    #   structural_level_constraints[constraint_subj] = constraint_subj_node
    #
    #   constraint_edge_node = constraint_subj_node.get(constraint_edge, {})
    #   constraint_subj_node[constraint_edge] = constraint_edge_node
    #
    #   constraint_node = constraint_edge_node.get(constraint_v, {})
    #   constraint_edge_node[constraint_v] = constraint_node
    #   constraint_edge_node[constraint_v]['value'] = v['value']
    #   constraint_edge_node[constraint_v]['span'] = v['span']
    #   constraint_edge_node[constraint_v]['confidence'] = v['confidence']

    # if v.get('parent'):
    #   tree[k] = 'done'
  clean_up_tree(tree)
  return tree


charter_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Charter",
  "description": "A charter document attributes",
  "type": "object",

  "definitions": {

    "tag": {
      "type": "object",
      "properties": {
        "_value": {},
        "span": {
          "type": "array",
          "minItems": 2,
          "maxItems": 2
        },
        "confidence": {
          "type": "number"
        }
      },
      "required": ["span", "_value"]
    },

    "structural_level":{
      "type": "object",
      "properties": {
        # "$ref": "#/definitions/tag",
        "_value": {
          "enum": OrgStructuralLevel._member_names_
        }
      }
    },

    "org": {
      "type": "object",
      "properties": {
        "name": {
          "$ref": "#/definitions/tag",
          "_value": {
            "type": "string"
          }
        },
        "alias": {
          "$ref": "#/definitions/tag",
          "_value": {
            "type": "string"
          }
        },
        "type": {
          "$ref": "#/definitions/tag",
          "_value": {
            "type": "string"
          }
        }
      }
    }
  },

  "properties": {
    "date": {
      "$ref": "#/definitions/tag",
      "properties": {
        "_value": {
          "type": "string",
          "format": "date-time"
        }
      }
    },

    "number": {
      "$ref": "#/definitions/tag",
      "properties": {
        "_value": {
          "type": "number"
        }
      }
    },

    "orgs": {
      "type": "array",
      "maxItems": 1,
      "items": {
        "$ref": "#/definitions/org",
      }
    },

    "structural_levels":{
      "type": "array",
      "items": {
        "$ref": "#/definitions/structural_level"
      }
    }
  }
}
if __name__ == '__main__':
  # charter: 5f64161009d100a445b7b0d6

  db = get_mongodb_connection()
  doc = get_doc_by_id(ObjectId('5f64161009d100a445b7b0d6'))
  a = doc['analysis']['attributes']
  tree = convert_charter_db_attributes_to_tree(a)

  # print(tree)
  # db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"analysis.attributes_tree": tree}})

  validate(instance=tree, schema=charter_schema)
