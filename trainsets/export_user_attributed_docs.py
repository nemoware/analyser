#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import json
import os

from bson import json_util

from analyser.hyperparams import work_dir
from integration.db import get_mongodb_connection


def get_user_updated_docs():
  print('obtaining DB connection...')
  db = get_mongodb_connection()
  print('obtaining DB connection: DONE')
  documents_collection = db['documents']
  print('linking documents collection: DONE')

  query = {"user.attributes": {"$ne": None}}

  print('running DB query')
  res = documents_collection.find(query)
  print('running DB query: DONE')
  docs = []
  print('reading docs....')

  for doc in res:
    docs.append(doc)

  print("---- Number of documents:", len(docs))

  return docs


def export_docs(documents):
  arr = []
  for k, d in enumerate(documents):

    if '_id' not in d['user']['author']:
      print(f'error: user attributes doc {d["_id"]} is not linked to any user')

    if 'auditId' not in d:
      print(f'error: doc {d["_id"]} is not linked to any audit')

    arr.append(d)
    print(k, d['_id'])

  with open(os.path.join(work_dir, 'exported_docs.json'), 'w', encoding='utf-8') as outfile:
    json.dump(arr, outfile, indent=2, ensure_ascii=False, default=json_util.default)


def export_user_attributed_docs():
  docs = get_user_updated_docs()
  export_docs(docs)


if __name__ == '__main__':
  # os.environ['GPN_DB_NAME'] = 'gpn'
  # os.environ['GPN_DB_HOST'] = '192.168.10.36'
  # os.environ['GPN_DB_PORT'] = '27017'
  db = get_mongodb_connection()

  export_user_attributed_docs()
