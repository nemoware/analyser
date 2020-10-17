import datetime
import json

from bson import json_util
from bson.objectid import ObjectId
from jsonschema import validate, FormatChecker

from analyser.finalizer import get_doc_by_id
from analyser.schemas import charter_schema
from analyser.structures import OrgStructuralLevel
from integration.db import get_mongodb_connection


def convert_org(attr_name: str, path: str, v, tree):
  orgs_arr = tree.get('orgs', [])
  tree['orgs'] = orgs_arr

  name_parts = attr_name.split('-')
  _index = int(name_parts[1]) - 1
  pad_array(orgs_arr, _index + 1)

  org_node = orgs_arr[_index]

  org_node[name_parts[-1]] = copy_attr(v, {})


def copy_attr(src, dest):
  for key in ['span', 'span_map', 'confidence']:
    dest[key] = src.get(key)
  dest['_value'] = src.get('value')
  return dest


def getput_node(subtree, key_name, defaultValue):
  _node = subtree.get(key_name, defaultValue)
  subtree[key_name] = _node  # put_it
  return _node


def pad_array(arr, size):
  for _i in range(len(arr), size):
    arr.append({})


def _find_or_put_by_value(arr: [], value):
  for c in arr:
    if c['_value'] == value:
      return c

  nv = {'_value': value}
  arr.append(nv)
  return nv


def get_or_create_constraint_node(path, tree):
  path_s: [] = path.split('/')

  # attr_name: str = path_s[-1]

  _structural_levels_arr = getput_node(tree, 'structural_levels', [])

  structural_level_node = _find_or_put_by_value(_structural_levels_arr, path_s[0])
  # getput_node(_structural_levels_dict, path_s[0], {})
  if len(path_s) == 1:
    return structural_level_node

  subjects_arr = getput_node(structural_level_node, 'competences', [])
  subject_node = _find_or_put_by_value(subjects_arr, path_s[1])

  if len(path_s) == 2:
    return subject_node

  constraint = path_s[2].split('-')
  # constraint_margin = constraint[1]  # min max
  constraint_margin_index = 0
  if len(constraint) == 3:
    constraint_margin_index = int(constraint[2]) - 1

  constraints_arr = getput_node(subject_node, "constraints", [])
  pad_array(constraints_arr, constraint_margin_index + 1)
  margin_node = constraints_arr[constraint_margin_index]
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
  pass

  for s_node in tree.get('structural_levels', []):
    for subj_node in s_node.get('competences', []):
      l = remove_empty_from_list(subj_node.get("constraints", []))
      subj_node['constraints'] = l


def convert_charter_db_attributes_to_tree(attrs):
  _index = {}
  tree = {}

  for path, v in attrs.items():
    key_s: [] = path.split('/')
    attr_name: str = key_s[-1]

    if v.get('span', None):  # only tags (leafs)

      # handle org
      if attr_name.startswith('org-'):
        convert_org(attr_name, path, v, tree)

      # handle date
      if (key_s[-1] == 'date'):
        tree[attr_name] = copy_attr(v, {})

      # handle constraints
      if (key_s[0] in OrgStructuralLevel._member_names_):
        convert_constraints(attr_name, path, v, tree)

  clean_up_tree(tree)
  return tree


if __name__ == '__main__':
  # charter: 5f64161009d100a445b7b0d6

  db = get_mongodb_connection()
  doc = get_doc_by_id(ObjectId('5f64161009d100a445b7b0d6'))
  a = doc['analysis']['attributes']
  tree = convert_charter_db_attributes_to_tree(a)

  # print(tree)

  def defaultd(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
      return o.astimezone().isoformat()
    return str(o)

  json_str=json.dumps(tree, default=defaultd)
  print(json_str)
  j = json.loads(json_str, object_hook=json_util.object_hook)
  validate(instance=j, schema=charter_schema, format_checker= FormatChecker())
  db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"analysis.attributes_tree": tree}})
