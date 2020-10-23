import datetime
import json

from bson import json_util
from bson.objectid import ObjectId
from jsonschema import validate, FormatChecker

from analyser.finalizer import get_doc_by_id
from analyser.schemas import  charter_schema
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


def index_of_key(s: str) -> int:
  n_i = s.split("-")
  _idx = 0
  if len(n_i) > 1:
    _idx = int(n_i[-1]) - 1
  return _idx


def convert_agenda_item(path, attr: {}, items_arr):
  _idx = index_of_key(path[0])

  pad_array(items_arr, _idx + 1)
  _item_node = items_arr[_idx]
  c_node = getput_node(_item_node, "contract", {})
  copy_attr(attr, _item_node)

  attr_name = path[-1]
  attr_name_parts = attr_name.split("-")
  attr_base_name = attr_name_parts[0]
  if "contract_agent_org" == attr_base_name:
    convert_org(attr_name, path[-1], attr, c_node)

  if len(path) == 2 and "sign_value_currency" == path[1]:
    v_node = getput_node(c_node, "value", {})
    copy_attr(attr, v_node)

  if len(path)==3 and "sign_value_currency"== path[1]:
    v_node = getput_node(c_node, "value", {})
    _pname = path[2]
    if _pname in ["value", "sign", "currency"]:
      v_node[_pname] =  copy_attr(attr, {})

  if (attr_base_name == 'date' or attr_base_name == 'number'):
    c_node[attr_base_name] = copy_attr(attr, {})
    if len(attr_name_parts) > 1:
      #add warning
      c_node['warnings'] = "ambigous values"
  pass


def convert_protocol_db_attributes_to_tree(attrs):
  tree = {}

  for path, v in attrs.items():
    key_s: [] = path.split('/')
    attr_name: str = key_s[-1]

    if v.get('span', None):  # only tags (leafs)

      # handle org
      if attr_name.startswith('org-'):
        convert_org(attr_name, path, v, tree)

      # handle date and number
      if (attr_name == 'date' or attr_name == 'number'):
        tree[attr_name] = copy_attr(v, {})

      if (attr_name == 'org_structural_level'):
        tree['structural_level'] = copy_attr(v, {})

      # handle agenda item
      root = key_s[0]
      root = root.split("-")
      if (root[0] == "agenda_item"):
        items_arr = getput_node(tree, "agenda_items", [])
        convert_agenda_item(key_s, v, items_arr)

  clean_up_tree(tree)
  return tree


def convert_charter_db_attributes_to_tree(attrs):
  tree = {}

  for path, v in attrs.items():
    key_s: [] = path.split('/')
    attr_name: str = key_s[-1]

    if v.get('span', None):  # only tags (leafs)

      # handle org
      if attr_name.startswith('org-'):
        convert_org(attr_name, path, v, tree)

      # handle date
      if ('date' == attr_name):
        tree[attr_name] = copy_attr(v, {})

      # handle constraints
      if (key_s[0] in OrgStructuralLevel._member_names_):
        convert_constraints(attr_name, path, v, tree)

  clean_up_tree(tree)
  return tree


if __name__ == '__main__':
  # charter: 5f64161009d100a445b7b0d6
  # protocol: 5ded4e214ddc27bcf92dd6cc
  # contract: 5f0bb4bd138e9184feef1fa8

  db = get_mongodb_connection()
  doc = get_doc_by_id(ObjectId('5ded4e214ddc27bcf92dd6cc'))
  a = doc['analysis']['attributes']
  # tree = {"charter": convert_charter_db_attributes_to_tree(a)}
  tree = {"protocol": convert_protocol_db_attributes_to_tree(a)}
  # tree = {"contract": convert_contract_db_attributes_to_tree(a)}


  # tree = convert_protocol_db_attributes_to_tree(a)

  # print(tree)

  def defaultd(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
      return o.astimezone().isoformat()
    return str(o)


  json_str = json.dumps(tree, default=defaultd)
  print(json_str)
  j = json.loads(json_str, object_hook=json_util.object_hook)
  validate(instance=j, schema=charter_schema, format_checker=FormatChecker())
  db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"analysis.attributes_tree": tree}})

  # coll = db["schemas"]
  # coll.delete_many({})
  # coll.insert_one( {"charter":charter_schema })

  # db.create_collection("test_charters", {"validator": {"$jsonSchema": charter_schema}})
