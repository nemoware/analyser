import os
import warnings

from pymongo import MongoClient

_db_client = None

def __check_var(vname):
  if vname not in os.environ:
    msg = f'MongoDB could not be connected, define {vname} environment variable'
    warnings.warn(msg)
    return False
  else:
    return True

# mongod --config /usr/local/etc/mongod.conf

def get_mongodb_connection():
  if __check_var('GPN_DB_HOST') and __check_var('GPN_DB_PORT') and __check_var('GPN_DB_NAME'):
    global _db_client
    if _db_client is None:
      _db_client = MongoClient(f'mongodb://{os.environ["GPN_DB_HOST"]}:{os.environ["GPN_DB_PORT"]}/')
    return _db_client[os.environ["GPN_DB_NAME"]]

  warnings.warn('DEFAULTING TO LOCAL MONGO')
  return _get_local_mongodb_connection()
  


def _get_local_mongodb_connection():
  global _db_client
  if _db_client is None:
    _db_client = MongoClient(f'mongodb://localhost:27017/')
  return _db_client['gpn']



if __name__ == '__main__':
  get_mongodb_connection()
