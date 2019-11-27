#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest


class TestAnalyse(unittest.TestCase):

  def test_on_smallText(self):
    microsample = """
    Общие положения

    1.1. В соответствии с настоящим Договором Жертвователь обязуется безвозмездно передать Получателю денежные средства в размере 30 000 (Тридцать тысяч) рублей в качестве пожертвования.

    """

    pass


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
