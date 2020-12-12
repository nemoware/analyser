#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import pymongo

from integration.db import get_mongodb_connection

if __name__ == '__main__':

  db = get_mongodb_connection()
  audits_collection = db['audits']

  cursor = audits_collection.find({'status': 'Done'}).sort([("createDate", pymongo.ASCENDING)])
  print("completed audits:")
  for audit in cursor:
    print(audit['_id'])
