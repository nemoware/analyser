#!/usr/bin/env python
import os
import time

import schedule

from analyser import runner
from analyser.dictionaries import update_db_dictionaries

import tensorflow as tf
def main():
  update_db_dictionaries()

  check_interval = os.environ.get("GPN_DB_CHECK_INTERVAL")
  if check_interval is None:
    check_interval = 30
    print("Environment variable GPN_DB_CHECK_INTERVAL not set. Default value is %d sec." % (check_interval))
  schedule.every(int(check_interval)).seconds.do(runner.run)

  runner.run()
  while True:
    schedule.run_pending()
    time.sleep(1)


if __name__ == '__main__':

  main()
