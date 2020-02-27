#!/usr/bin/env python
import os
import time

import schedule

import analyser
from analyser import runner
from gpn.gpn import update_subsidiaries_in_db


def main():
  update_subsidiaries_in_db()
  check_interval = os.environ.get("GPN_DB_CHECK_INTERVAL")
  if check_interval is None:
    check_interval = 30
    print("Environment variable GPN_DB_CHECK_INTERVAL not set. Default value is %d sec." %(check_interval))
  schedule.every(int(check_interval)).seconds.do(runner.run)

  runner.run()
  while True:
    schedule.run_pending()
    time.sleep(1)


if __name__ == '__main__':
  main()