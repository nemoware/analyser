import os
import warnings

from pymongo import MongoClient

_db_client = None
_db_name = None


def __check_var(vname):
  if vname not in os.environ:
    msg = f'MongoDB could not be connected, define {vname} environment variable'
    warnings.warn(msg)
    return False
  else:
    return True


# mongod --config /usr/local/etc/mongod.conf

def get_mongodb_connection():
  global _db_client
  global _db_name
  if _db_client is None:
    try:
      if __check_var('GPN_DB_HOST') and __check_var('GPN_DB_PORT') and __check_var('GPN_DB_NAME'):
        _db_name = os.environ["GPN_DB_NAME"]
        _db_client = MongoClient(f'mongodb://{os.environ["GPN_DB_HOST"]}:{os.environ["GPN_DB_PORT"]}/')
        _db_client.server_info()
        return _db_client[_db_name]
    except Exception as err:
      _db_client = None
      warnings.warn(err)
  else:
    return _db_client[_db_name]

  warnings.warn('defaulting MongoDB to mongodb://localhost:27017/')
  return _get_local_mongodb_connection()


def _get_local_mongodb_connection():
  global _db_client
  global _db_name
  _db_name = 'gpn'
  if _db_client is None:
    try:
      _db_client = MongoClient(f'mongodb://localhost:27017/')
      _db_client.server_info()
      return _db_client[_db_name]
    except Exception as err:
      _db_client = None
      msg=f'{err}'
      warnings.warn(msg)
  else:
    return _db_client[_db_name]
  return None


if __name__ == '__main__':
  get_mongodb_connection()
