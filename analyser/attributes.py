import json
from datetime import datetime, date

import jsonpickle
from bson import json_util
from bson.objectid import ObjectId
from jsonschema import validate, FormatChecker

from analyser.finalizer import get_doc_by_id
from analyser.log import logger
from analyser.ml_tools import SemanticTagBase
from analyser.schemas import charter_schema, ProtocolSchema, OrgItem, AgendaItem, AgendaItemContract, HasOrgs, \
  ContractPrice, ContractSchema
from analyser.structures import OrgStructuralLevel
from integration.db import get_mongodb_connection


def convert_org(attr_name: str, attr: dict, dest: HasOrgs):
  orgs_arr = dest.orgs

  name_parts = attr_name.split('-')
  _index = int(name_parts[1]) - 1
  pad_array(orgs_arr, _index + 1, OrgItem())

  org = orgs_arr[_index]

  field_name = name_parts[-1]

  if field_name in ['type', 'name', 'alias']:
    copy_leaf_tag(field_name, src=attr, dest=org)


def copy_leaf_tag(field_name: str, src, dest: SemanticTagBase):
  if hasattr(dest, field_name):
    v = getattr(dest, field_name)
    setattr(v, "warning", "ambiguity: multiple values, see 'alternatives' field")
    logger.warning(f"{field_name} has multiple values")

    alternatives = []
    if not hasattr(v, "alternatives"):
      setattr(v, "alternatives", alternatives)
    else:
      alternatives = getattr(v, "alternatives")

    va = SemanticTagBase()

    copy_attr(src, va)
    alternatives.append(va)

  else:
    v = SemanticTagBase()
    setattr(dest, field_name, v)

    copy_attr(src, v)


def copy_attr(src, dest: SemanticTagBase) -> SemanticTagBase:
  for key in ['span', 'span_map', 'confidence', "value"]:
    setattr(dest, key, src.get(key))
  #   dest[key] = src.get(key)
  # dest['_value'] = src.get('value')
  return dest


def getput_node(subtree, key_name, defaultValue):
  _node = subtree.get(key_name, defaultValue)
  subtree[key_name] = _node  # put_it
  return _node


def pad_array(arr, size, pad_with=None):
  for _i in range(len(arr), size):
    if pad_with is None:
      pad_with = {}
    arr.append(pad_with)


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


def index_of_key(s: str) -> (str, int):
  n_i = s.split("-")
  _idx = 0
  if len(n_i) > 1:
    _idx = int(n_i[-1]) - 1
  return n_i[0], _idx


def convert_agenda_item(path, attr: {}, _item_node: AgendaItem):
  '''

  :param path: like 'agenda_item-1/contract_agent_org-1-type'
  :param attr:
  :param _item_node:
  :return:
  '''
  if _item_node.contract is None:
    _item_node.contract = AgendaItemContract()

  c_node: AgendaItemContract = _item_node.contract

  attr_name = path[-1]  # 'contract_agent_org-1-type'
  attr_name_parts = attr_name.split("-")
  attr_base_name = attr_name_parts[0]  # 'contract_agent_org'
  if "contract_agent_org" == attr_base_name:
    convert_org(attr_name, attr=attr, dest=c_node)

  if len(path) > 1 and "sign_value_currency" == path[1]:
    convert_sign_value_currency(path[1:], attr, c_node)

  if attr_base_name in ['date', 'number']:
    copy_leaf_tag(attr_base_name, src=attr, dest=c_node)

  # pass


def map_tag(src, dest: SemanticTagBase = None) -> SemanticTagBase:
  if dest is None:
    dest = SemanticTagBase()

  _fields = ['value', 'span', 'confidence']
  for f in _fields:
    setattr(dest, f, src.get(f))

  return dest


def map_org(attr_name: str, v, dest: OrgItem) -> OrgItem:
  name_parts = attr_name.split('-')
  _index = int(name_parts[1]) - 1
  _field = name_parts[-1]
  if _field in ['name', 'type']:
    setattr(dest, _field, map_tag(v))

  return dest


def list_tags(attrs):
  for path, v in attrs.items():
    key_s: [] = path.split('/')

    if v.get('span', None):
      yield key_s, v


def convert_contract_db_attributes_to_tree(attrs) -> ContractSchema:
  tree = ContractSchema()
  for path, v in list_tags(attrs):
    attr_name: str = path[-1]

    # handle date and number
    if attr_name in ['date', 'number', 'subject']:
      copy_leaf_tag(attr_name, v, tree)

    if attr_name.startswith('org-'):
      convert_org(attr_name, attr=v, dest=tree)

    if "sign_value_currency" == path[0]:
      convert_sign_value_currency(path, v, tree)

  return tree


def convert_sign_value_currency(path, v, tree):
  if tree.price is None:
    tree.price = ContractPrice()

  if len(path) == 1:
    copy_attr(v, tree.price)

  if len(path) == 2:
    _pname = path[1]

    if _pname in ["sign", "currency"]:
      copy_leaf_tag(_pname, v, tree.price)
    if _pname == "value":
      copy_leaf_tag('amount', v, tree.price)


def convert_protocol_db_attributes_to_tree(attrs) -> ProtocolSchema:
  tree = ProtocolSchema()

  # collect agenda_items roots
  for key_s, v in list_tags(attrs):

    attr_name: str = key_s[-1]
    attr_name_clean = attr_name.split("-")
    if ("agenda_item" == attr_name_clean[0]):
      tree.agenda_items.append(map_tag(v, AgendaItem()))

  for key_s, v in list_tags(attrs):
    attr_name: str = key_s[-1]

    if v.get('span', None):  # only tags (leafs)

      # handle org
      if attr_name.startswith('org-'):
        tree.org = map_org(attr_name, v, tree.org)

      # handle date and number
      if (attr_name == 'date'):
        tree.date = map_tag(v)

      if (attr_name == 'number'):
        tree.number = map_tag(v)

      if (attr_name == 'org_structural_level'):
        tree.structural_level = map_tag(v)

      # handle agenda item
      root = key_s[0].split('-')
      # _i, _n = index_of_key( key_s[0])
      # root = root.split("-")
      if (root[0] == "agenda_item"):
        _, _i = index_of_key(key_s[0])
        convert_agenda_item(key_s, v, tree.agenda_items[_i])
  #
  # clean_up_tree(tree)
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
  doc = get_doc_by_id(ObjectId('5ded4e214ddc27bcf92dd6cc'))  # protocol
  # doc = get_doc_by_id(ObjectId('5f0bb4bd138e9184feef1fa8'))  # contract
  # a = doc['user']['attributes']
  a = doc['analysis']['attributes']
  # tree = {"charter": convert_charter_db_attributes_to_tree(a)}
  tree = convert_protocol_db_attributes_to_tree(a)


  # tree = convert_contract_db_attributes_to_tree(a)

  class DatetimeHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj: datetime, data):
      return json_util.default(obj)


  jsonpickle.handlers.registry.register(datetime, DatetimeHandler)
  jsonpickle.handlers.registry.register(date, DatetimeHandler)

  json_str = jsonpickle.encode(tree, unpicklable=False, indent=4)

  print(json_str)

  j = json.loads(json_str, object_hook=json_util.object_hook)
  j = {"protocol": j}
  validate(instance=json_str, schema=charter_schema, format_checker=FormatChecker())
  db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"analysis.attributes_tree": j}})

  # coll = db["schemas"]
  # coll.delete_many({})
  # coll.insert_one( {"charter":charter_schema })

  # db.create_collection("test_charters", {"validator": {"$jsonSchema": charter_schema}})
