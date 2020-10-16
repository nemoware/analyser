from bson.objectid import ObjectId

from analyser.finalizer import get_doc_by_id
from analyser.structures import OrgStructuralLevel
from integration.db import get_mongodb_connection


def convert_charter_db_attributes_to_tree(attrs):
  _index = {}

  structural_levels_dict = {}
  for path, v in attrs.items():
    key_s: [] = path.split('/')
    print(key_s[-1], '\t\t', path, v)

    root_s = key_s[0]
    if (len(key_s) == 4 and root_s in OrgStructuralLevel._member_names_):

      structural_level_constraints = structural_levels_dict.get(root_s, {})
      constraint_subj = key_s[1]
      constraint_edge = key_s[2].split('-')[1]

      constraint_v = key_s[3]
      structural_levels_dict[root_s] = structural_level_constraints

      constraint_subj_node = structural_level_constraints.get(constraint_subj, {})
      structural_level_constraints[constraint_subj] = constraint_subj_node

      constraint_edge_node = constraint_subj_node.get(constraint_edge, {})
      constraint_subj_node[constraint_edge] = constraint_edge_node

      constraint_node = constraint_edge_node.get(constraint_v, {})
      constraint_edge_node[constraint_v] = constraint_node
      constraint_edge_node[constraint_v]['value'] = v['value']
      constraint_edge_node[constraint_v]['span'] = v['span']
      constraint_edge_node[constraint_v]['confidence'] = v['confidence']


      # if v.get('parent'):
      #   tree[k] = 'done'

  return tree


if __name__ == '__main__':
  # charter: 5f64161009d100a445b7b0d6

  db = get_mongodb_connection()

  doc = get_doc_by_id(ObjectId('5f64161009d100a445b7b0d6'))
  a = doc['analysis']['attributes']
  tree = convert_charter_db_attributes_to_tree(a)
  # print(a)
  print(tree)
