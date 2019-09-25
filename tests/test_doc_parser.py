#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import csv
import sys
import traceback
import unittest
# pprint library is used to make the output look more pretty
from pprint import pprint
from typing import List

from pymongo import MongoClient

from contract_agents import find_org_names
from integration.word_document_parser import WordDocParser, join_paragraphs
from legal_docs import LegalDocument, Paragraph, DocumentJson
from ml_tools import SemanticTag
from text_normalize import normalize_text, replacements_regex


class TestContractParser(unittest.TestCase):

  def n(self, txt):
    return normalize_text(txt, replacements_regex)

  def ___test_list_files(self):
    wp = WordDocParser()
    filenames = wp.list_filenames('/Users/artem/Downloads/Telegram Desktop/X0/')
    cnt = 0
    for fn in filenames:
      cnt += 1
      shortfn = fn.split('/')[-1]
      if shortfn.lower().find('договор') >= 0:
        print(cnt, shortfn)

  def _trim_and_pad(l):
    return (l + [''] * 40)[0:10]  # padding: make all linese 10 columns-long

  def export_csv(self, rows, headline=['1', '2', '3', '4', '5', '6', '7', '8', '9']):
    with open(f'contracts-stats.csv', mode='w') as csv_file:
      _writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      _writer.writerow(headline)
      for l in rows:
        _writer.writerow(l)

  def ___test_list_wrong_docs(self):

    def _find_tag(name, tags):
      for t in tags:
        if t['kind'] == name:
          return t
      return None

    client = MongoClient('mongodb://localhost:27017/')
    db = client['docsdatabase']
    collection = db['contracts']

    wp = WordDocParser()
    filenames = wp.list_filenames('/Users/artem/Downloads/Telegram Desktop/X0/')
    cnt = 0
    for fn in filenames:

      shortfn = fn.split('/')[-1]
      pth = '/'.join(fn.split('/')[5:-1])
      _doc_id = pth + '/' + shortfn

      res = collection.find_one({"_id": _doc_id})
      if res is not None and 'tags' in res:
        if _find_tag('org.3.name', res['tags']) is not None:
          cnt += 1
          print(f'{cnt}\t{_doc_id}')

  def test_parse_ALL_docs(self):

    client = MongoClient('mongodb://localhost:27017/')
    db = client['docsdatabase']
    collection = db['legaldocs']
    contracts_collection = db['contracts']
    wp = WordDocParser()
    filenames = wp.list_filenames('/Users/artem/Downloads/Telegram Desktop/X0/')

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
          contract = self._parse_contract(res, _doc_id, row)
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
        self.export_csv(rows)
    stats()
    # print(f'failures:\t{failures}\t unknowns: {unknowns}\t nodate: {nodate}' )

  def _tag_to_atributes(self, tags: List[SemanticTag]):
    cnt = 0
    atributes = {}
    for t in tags:
      cnt += 1
      key = t.kind.replace('.', '_')
      if key in atributes:
        key = f'{key}_{cnt}'
      atributes[key] = t.__dict__.copy()
      del atributes[key]['kind']

    return atributes

  def _parse_contract(self, res, doc_id, row):
    contract = join_paragraphs(res, doc_id)
    # agent_infos = find_org_names_spans(contract.tokens_map_norm)
    # contract.agents_tags = agent_infos_to_tags(agent_infos)
    contract.agents_tags = find_org_names(contract)
    row[4:8] = [contract.tag_value('org.1.name'),
                contract.tag_value('org.1.alias'),
                contract.tag_value('org.2.name'),
                contract.tag_value('org.2.alias')]
    return contract

  def test_mongodb_connection(self):
    # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
    client = MongoClient('mongodb://localhost:27017/')
    db = client['docsdatabase']
    # Issue the serverStatus command and print the results
    serverStatusResult = db.command("serverStatus")
    pprint(serverStatusResult)

  def test_doc_parser(self):
    FILENAME = "/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx"

    wp = WordDocParser()
    res = wp.read_doc(FILENAME)

    doc: LegalDocument = LegalDocument('')
    doc.parse()

    last = 0
    for p in res['paragraphs']:
      header_text = p['paragraphHeader']['text'] + '\n'
      body_text = p['paragraphBody']['text'] + '\n'

      header = LegalDocument(header_text)
      header.parse()
      # self.assertEqual(self.n(header_text), header.text)

      doc += header
      headerspan = (last, len(doc.tokens_map))
      print(headerspan)
      last = len(doc.tokens_map)

      body = LegalDocument(body_text)
      body.parse()
      doc += body
      bodyspan = (last, len(doc.tokens_map))

      header_tag = SemanticTag('headline', header_text, headerspan)
      body_tag = SemanticTag('paragraphBody', None, bodyspan)

      print(header_tag)
      # print(body_tag)
      para = Paragraph(header_tag, body_tag)
      doc.paragraphs.append(para)
      last = len(doc.tokens_map)

      h_subdoc = doc.subdoc_slice(para.header.as_slice())
      b_subdoc = doc.subdoc_slice(para.body.as_slice())
      # self.assertEqual(self.n(header_text), h_subdoc.text)
      # self.assertEqual(self.n(body_text), b_subdoc.text)

    print('-' * 100)
    print(doc.text)

    headers = [doc.subdoc_slice(p.header.as_slice()) for p in doc.paragraphs]
    print('-' * 100)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
