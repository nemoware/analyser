#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import csv
import json
import sys
import traceback

from pymongo import MongoClient

from contract_agents import find_org_names
from integration.word_document_parser import WordDocParser, join_paragraphs
from legal_docs import DocumentJson

files_dir = '/Users/artem/Downloads/Telegram Desktop/X0/'


def read_all_contracts():
  client = MongoClient('mongodb://localhost:27017/')
  db = client['docsdatabase']
  collection = db['legaldocs']
  contracts_collection = db['contracts']
  wp = WordDocParser()
  filenames = wp.list_filenames(files_dir)

  cnt = 0
  failures = 0
  unknowns = 0
  nodate = 0

  rows = []

  def stats():
    print(f'processed:{cnt};\t failures:\t{failures}\t unknown type: {unknowns}\t unknown date: {nodate}')

  for fn in filenames:

    shortfn = fn.split('/')[-1]
    pth = '/'.join(fn.split('/')[5:-1])
    _doc_id = pth + '/' + shortfn

    cnt += 1
    print(cnt, fn)

    res = collection.find_one({"_id": _doc_id})
    if res is None:
      try:
        # parse and save to DB
        res = wp.read_doc(fn)

        res['short_filename'] = shortfn
        res['path'] = pth
        res['_id'] = _doc_id

        collection.insert_one(res)
        # print('post_id-----------', post_id)
        # doc = join_paragraphs(res)

      except Exception:
        print(f"{fn}\nException in WordDocParser code:")
        traceback.print_exc(file=sys.stdout)
        failures += 1
        err = True
        res = None
    else:
      print(cnt, res["documentDate"], res["documentType"])

    row = [cnt, shortfn, None, None, None, None, None, None, pth, None]
    if res:
      row[2:4] = [res["documentType"], res["documentDate"]]

      if 'UNKNOWN' == res["documentType"]:
        unknowns += 1
        stats()

      if res["documentDate"] is None:
        nodate += 1
        stats()

      if 'CONTRACT' == res["documentType"]:
        contract = _parse_contract(res, _doc_id, row)
        json_struct = DocumentJson(contract).__dict__
        # json_struct['_id'] = _doc_id
        #
        # if res["documentDate"]:
        #   date_time_obj = datetime.datetime.strptime(res["documentDate"], '%Y-%m-%d')
        #   json_struct['attributes']['documentDate'] = {
        #     'value': date_time_obj,
        #     'display_value': res["documentDate"]
        #   }
        #

        if contracts_collection.find_one({"_id": _doc_id}) is None:
          contracts_collection.insert_one(json_struct)

        # else:
        #   contracts_collection.update_one({'$setOnInsert':json_struct})

    rows.append(row)

    if cnt % 20 == 0:
      # print(rows)
      export_csv(rows)
  stats()
  # print(f'failures:\t{failures}\t unknowns: {unknowns}\t nodate: {nodate}' )


def _parse_contract(res, doc_id, row):
  contract = join_paragraphs(res, doc_id)

  contract.agents_tags = find_org_names(contract)
  row[4:8] = [contract.tag_value('org.1.name'),
              contract.tag_value('org.1.alias'),
              contract.tag_value('org.2.name'),
              contract.tag_value('org.2.alias')]
  return contract


def export_csv(rows, headline=['1', '2', '3', '4', '5', '6', '7', '8', '9']):
  with open(f'{files_dir}contracts-stats.csv', mode='w') as csv_file:
    _writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    _writer.writerow(headline)
    for l in rows:
      _writer.writerow(l)


def dump_contracts_from_db_to_jsons(output_path):

  client = MongoClient('mongodb://localhost:27017/')
  db = client['docsdatabase']
  collection = db['legaldocs']

  wp = WordDocParser()
  filenames = wp.list_filenames('/Users/artem/Downloads/Telegram Desktop/X0/')

  for fn in filenames:
    print(fn)
    shortfn = fn.split('/')[-1]
    pth = '/'.join(fn.split('/')[5:-1])
    _doc_id = pth + '/' + shortfn

    res = collection.find_one({"_id": _doc_id})
    if res is not None:
      json_name = _doc_id.replace('/','_')
      with open(f'{output_path}/{json_name}.json', 'w') as file:
        _j = json.dumps(res, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')
        file.write(_j)
        print(f'saved file to {json_name}')

if __name__ == '__main__':
  # read_all_contracts()
  dump_contracts_from_db_to_jsons('/Users/artem/work/nemo/goil/OUT/jsons')
