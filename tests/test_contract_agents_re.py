#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_agents import *
from contract_agents import _find_org_names
from contract_augmentation import augment_contract
from text_normalize import _r_name_ru, r_human_abbr_name, r_human_full_name, _r_name_lat

find_org_names = _find_org_names

def n(x):
  return normalize_contract(x )

class TestTextNormalization(unittest.TestCase):
  # def _testr(self, r, str, expected):
  #

  def test_ru_cap(self):
    x = ru_cap(n('Государственной автономной учрежденией'))
    self.assertEqual('[Гг]осударственн[а-я]{0,3}\s+[Аа]втономн[а-я]{0,3}\s+[Уу]чреждени[а-я]{0,3}', x)

    x = ru_cap('автономной учрежденией')
    self.assertEqual('[Аа]втономн[а-я]{0,3}\s+[Уу]чреждени[а-я]{0,3}', x)

  def test_r_name(self):

    r = re.compile(_r_name_ru, re.MULTILINE)
    x = r.search('УУУ')
    print(x)
    x = r.search('ННН')
    print(x)

    r = re.compile(_r_name_lat, re.MULTILINE)
    x = r.search('YYy')
    print(x)

  def test_r_type_and_name(self):

    r = re.compile(r_type_and_name, re.MULTILINE)

    x = r.search(n('Общество с ограниченной ответственностью « Газпромнефть-Региональные продажи » и вообще'))
    self.assertEqual('Общество с ограниченной ответственностью', x['type'])
    self.assertEqual('Газпромнефть-Региональные продажи', x['name'])

    x = r.search(n('Общество с ограниченной ответственностью и прочим « Газпромнефть-Региональные продажи » и вообще'))
    self.assertEqual(x[1], 'Общество с ограниченной ответственностью')
    self.assertEqual(x[5], 'Газпромнефть-Региональные продажи')

    x = r.search(n('Общество с ограниченной ответственностью и прочим «Меццояха» и вообще'))
    self.assertEqual(x['type'], 'Общество с ограниченной ответственностью')
    self.assertEqual(x['name'], 'Меццояха')

    x = r.search('с ООО «УУУ»')
    self.assertEqual(x['type'], 'ООО')
    self.assertEqual(x[5], 'УУУ')

    x = r.search(n('с большой Акционерной обществой  « УУУ » и вообще'))
    self.assertEqual('Акционерной обществой', x[1])
    self.assertEqual('УУУ', x[5])

    x = r.search(n('с большой Государственной автономной учрежденией  « УУУ »'))
    self.assertEqual('УУУ', x[5])
    self.assertEqual('Государственной автономной учрежденией', x[1])

    t = """
       ООО «Газпромнефть-Региональные продажи» в дальнейшем «Благотворитель», с другой стороны
       """
    x = r.search(t)
    for c in range(6):
      print(c, x[c])
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Газпромнефть-Региональные продажи', x['name'])

  def test_r_type_and_name_2(self):

    r = re.compile(r_type_and_name, re.MULTILINE)

    t = """
       ООО «Газпромнефть-Региональные продажи», в дальнейшем «Благотворитель», с другой стороны
       """
    x = r.search(t)
    for c in range(6):
      print(c, x[c])
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Газпромнефть-Региональные продажи', x['name'])

  def test_org_re(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    x = r.search(
      'некое Общество с ограниченной ответственностью «Меццояхве » ( или "Иначе"), именуемое в дальнейшем "Нечто"')
    for c in range(13):
      print(c, x[c])

    self.assertEqual('Общество с ограниченной ответственностью', x[1])
    self.assertEqual('Меццояхве', x['name'])
    self.assertEqual('( или "Иначе")', x['alt_name'])
    self.assertEqual('Нечто', x['alias'])

  def test_org_re_2(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    x = r.search(
      'некое Общество с ограниченной ответственностью «НЕ БЕЗ НАЗВАНИЯ» ( или "Иначе"), как бе именуемое в дальнейшем "Как-то так"')
    # for c in range(13):
    #   print(c, x[c])

    self.assertEqual('Общество с ограниченной ответственностью', x[1])
    self.assertEqual('НЕ БЕЗ НАЗВАНИЯ', x['name'])
    self.assertEqual('( или "Иначе")', x['alt_name'])
    self.assertEqual('Как-то так', x['alias'])

  def test_org_dict_1(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    t = n("""
    ООО «Газпромнефть-Региональные продажи», в лице начальника управления по связям с общественностью Иванова Семена Евгеньевича, действующего на основании Доверенности в дальнейшем «Благотворитель», с другой стороны заключили настоящий Договор о
    нижеследующем:
    """)
    #
    x = r.search(t)
    for c in range(16):
      print(c, x[c])

    onames = find_org_names(t)
    print(onames[0])
    r = onames[0]
    self.assertEqual('ООО', r['type'][0])
    self.assertEqual('Газпромнефть-Региональные продажи', r['name'][0])
    self.assertEqual('Благотворитель', r['alias'][0])

  def test_org_dict_2(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    t = n("""
    ООО «Газпромнефть-Региональные продажи» в дальнейшем «Благотворитель», с другой стороны
    """)

    r1 = re.compile(r_quoted_name, re.MULTILINE)
    x1 = r1.search(t)
    print(x1['name'])
    # for c in range(16):
    #   print(c, x1[c])

    #
    x = r.search(t)
    for c in range(16):
      print(c, x[c])

    onames = find_org_names(t)
    print(onames[0])
    r = onames[0]
    self.assertEqual('ООО', r['type'][0])
    self.assertEqual('Газпромнефть-Региональные продажи', r['name'][0])
    self.assertEqual('Благотворитель', r['alias'][0])

  def test_org_dict_3(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    t = n("""
    Муниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем
    «Благополучатель», в лице директора Соляной Марины Александровны, действующая на основании
    Устава, с одной стороны, и ООО «Газпромнефть-Региональные продажи», аааааааа аааа в дальнейшем «Благотворитель», с другой стороны заключили настоящий Договор о
    нижеследующем:
    """)
    #
    x = r.search(t)

    #
    # self.assertEqual('Муниципальное бюджетное учреждение', x[1])
    # self.assertEqual('Радуга', x[5])
    # # self.assertEqual('( или "Иначе")', x[7])
    # self.assertEqual('Благополучатель', x[15])

    onames = find_org_names(t)
    print(onames[0])
    r = onames[0]
    self.assertEqual('Муниципальное бюджетное учреждение', r['type'][0])
    self.assertEqual('Радуга', r['name'][0])
    self.assertEqual('Благополучатель', r['alias'][0])

    # onames = find_org_names(t[200:])
    print(onames[1])
    r = onames[1]
    self.assertEqual('ООО', r['type'][0])
    self.assertEqual('Газпромнефть-Региональные продажи', r['name'][0])
    self.assertEqual('Благотворитель', r['alias'][0])

  def test_org_dict_4(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    t = n("""
    Сибирь , и Индивидуальный предприниматель « Петров В. В. » , именуемый в дальнейшем « Исполнитель » , с другой стороны , именуемые в дальнейшем совместно « Стороны » , а по отдельности - « Сторона » , заключили настоящий договор о нижеследующем : 
    """)

    onames = find_org_names(t)
    print(onames[0])
    r = onames[0]
    self.assertEqual('Индивидуальный предприниматель', r['type'][0])
    self.assertEqual('Петров В. В.', r['name'][0])
    self.assertEqual('Исполнитель', r['alias'][0])

    t = n("""Автономная некоммерческая организация дополнительного профессионального образования «ООО», именуемое далее Исполнитель, в лице Директора Уткиной Е.В., действующей на основании Устава, с одной стороны,""")
    onames = find_org_names(t)
    print(onames[0])
    r = onames[0]
    self.assertEqual('некоммерческая организация', r['type'][0])
    self.assertEqual('ООО', r['name'][0])
    # self.assertEqual('Исполнитель', r['alias'][0])

    t = n("""Государственное автономное  учреждение дополнительного профессионального образования Свердловской области «Армавирский учебно-технический центр»,  на основании Лицензии на право осуществления образовательной деятельности в лице директора  Птицына Евгения Георгиевича, действующего на основании Устава, с одной стороны, """)
    onames = find_org_names(t)
    print(onames[0])
    r = onames[0]
    self.assertEqual('Государственное автономное учреждение', r['type'][0])
    self.assertEqual('Армавирский учебно-технический центр', r['name'][0])

  # def test_augment_contract(self):
  #   t = """
  #       Муниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем
  #       «Благополучатель», в лице директора Соляной Марины Александровны, действующая на основании
  #       Устава, с одной стороны, и
  #
  #       ООО «Газпромнефть-Региональные продажи» в лице начальника управления по связям с общественностью Иванова Семена Евгеньевича, действующего на основании Доверенности в дальнейшем «Благотворитель», с другой стороны заключили настоящий Договор о нижеследующем:
  #       """
  #   onames = find_org_names(t)
  #   x,y = augment_contract(t, onames)
  #   print(x, y)

  def test_org_dict(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    t = """
    Муниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем
    «Благополучатель», в лице директора Соляной Марины Александровны, действующая на основании
    Устава, с одной стороны, и 
    
    ООО «Газпромнефть-Региональные продажи» в лице начальника управления по связям с общественностью Иванова Семена Евгеньевича, действующего на основании Доверенности в дальнейшем «Благотворитель», с другой стороны заключили настоящий Договор о нижеследующем:
    """
    #
    x = r.search(t)

    #
    # self.assertEqual('Муниципальное бюджетное учреждение', x[1])
    # self.assertEqual('Радуга', x[5])
    # # self.assertEqual('( или "Иначе")', x[7])
    # self.assertEqual('Благополучатель', x[15])

    onames = find_org_names(t)
    print(onames[0])
    r = onames[0]
    self.assertEqual('Муниципальное бюджетное учреждение', r['type'][0])
    self.assertEqual('Радуга', r['name'][0])
    self.assertEqual('Благополучатель', r['alias'][0])

    # onames = find_org_names(t[200:])
    print(onames[1])
    r = onames[1]
    self.assertEqual('ООО', r['type'][0])
    self.assertEqual('Газпромнефть-Региональные продажи', r['name'][0])
    self.assertEqual('Благотворитель', r['alias'][0])

    # //self.assertEqual(x[0], 'Общество с ограниченной ответственностью')

  def test_r_types(self):
    r = re.compile(r_types, re.MULTILINE)

    x = r.search('с большой Общество с ограниченной ответственностью « Газпромнефть-Региональные продажи »')
    self.assertEqual(x[0], 'Общество с ограниченной ответственностью')

    x = r.search('Общество с ограниченной ответственностью и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual(x[0], 'Общество с ограниченной ответственностью')

    x = r.search('ООО « Газпромнефть-Региональные продажи »')
    self.assertEqual(x[0], 'ООО')

    x = r.search('с большой Государственной автономной учрежденией')
    self.assertEqual(x[0], 'Государственной автономной учрежденией')



    x = r.search('акционерное Общество с ограниченной ответственностью и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual(x[0], 'акционерное Общество')

    x = r.search('АО  и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual(x[0], 'АО')

    x = r.search('префикс АО  и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual(x[0], 'АО')

  def test_r_alias_prefix(self):
    r = re.compile(r_alias_prefix, re.MULTILINE)
    print(r)

    x = r.search('что-то именуемое в дальнейшем Жертвователь, и ')
    self.assertEqual(x[2], 'именуемое в дальнейшем ')

    x = r.search('именуемое далее Жертвователь или еще как')
    self.assertEqual(x[2], 'именуемое далее ')

    x = r.search('в дальнейшем  Жертвователь или еще как')
    self.assertEqual(x[2], 'в дальнейшем ')

    x = r.search('нечто, именуемое далее Жертвователь или еще как')
    self.assertEqual(x[2], 'именуемое далее ')

    x = r.search('нечто, далее - Жертвователь или еще как')
    self.assertEqual(x[1], ' далее -')

  def test_r_alias(self):
    r = re.compile(r_alias, re.MULTILINE)

    x = r.search('что-то, именуемое в дальнейшем « Жертвователь » или иначе')
    self.assertEqual('Жертвователь', x['alias'])

    x = r.search('что-то именуемое в дальнейшем «Жертвователь-какаха», и ')
    self.assertEqual('Жертвователь-какаха', x['alias'])

  def test_r_quoted_name(self):
    rgc = re.compile(r_quoted_name_alias, re.MULTILINE)

    x = rgc.search('что-то именуемое в дальнейшем " Абралябмда филорна", и ')
    self.assertEqual('Абралябмда филорна', x['alias'])

    x = rgc.search('что-то именуемое в дальнейшем «Жертвователь», и ')
    self.assertEqual('Жертвователь', x['alias'])

    t = n("""
        ООО «Газпромнефть-Региональные продажи» в дальнейшем «Благотворитель», с другой стороны
        """)

    r1 = re.compile(r_quoted_name, re.MULTILINE)
    x1 = r1.search(t)

    self.assertEqual('Газпромнефть-Региональные продажи', x1['name'])

  def test_replace_alias(self):
    r = sub_alias_quote

    t = 'что-то именуемое в дальнейшем Жертвователь-какаха, и нечто, именуемое далее КАКАХА-ХА '

    x = sub_alias_quote[0].search(t)
    for c in range(7):
      print(c, x[c])

    replacer = r[1]
    pattern = r[0]

    xx = pattern.sub(replacer, t)
    self.assertEqual('что-то именуемое в дальнейшем «Жертвователь-какаха», и нечто, именуемое далее «КАКАХА-ХА» ', xx)

    xx = pattern.sub(replacer, 'что-то в дальнейшем Жертвователь-какаха, и нечто, именуемое далее КАКАХА-ХА')
    self.assertEqual('что-то в дальнейшем «Жертвователь-какаха», и нечто, именуемое далее «КАКАХА-ХА»', xx)

    xx = pattern.sub(replacer, 'что-то, далее - Благожертвователь-какаха, и нечто')
    self.assertEqual('что-то, далее - «Благожертвователь-какаха», и нечто', xx)

    xx = pattern.sub(replacer, 'далее - Благожертвователь-какаха, и нечто')
    self.assertEqual('далее - «Благожертвователь-какаха», и нечто', xx)

  def test_r_human_full_name(self):
    r = re.compile(r_human_full_name, re.MULTILINE)

    x = r.search('что-то Мироздания Леонидыч Крупица, Который "был" таков')
    self.assertEqual('Мироздания Леонидыч Крупица', x[1])

    x = r.search('что-то Мироздания Крупица, который был')
    self.assertEqual('Мироздания Крупица', x[1])

    x = r.search('Мироздания Крупица , который был')
    self.assertEqual('Мироздания Крупица', x[1])

    x = r.search('что-то Абрам, который был')
    self.assertEqual(None, x)

  def test_r_human_abbr_name(self):
    r = re.compile('\W' + r_human_abbr_name, re.MULTILINE)

    x = r.search('что-то Мироздания С.К., который был')
    self.assertEqual('Мироздания С.К.', x[1])

    x = r.search('что-то Мироздания С. К., который был')
    self.assertEqual('Мироздания С. К.', x[1])

    x = r.search('что-то Мироздания С. , который был')
    self.assertEqual('Мироздания С. ', x[1])

    x = r.search('что-то  С. , который был')
    self.assertEqual(None, x)

    x = r.search('что-то друГое  С. R. , который был')
    self.assertEqual(None, x)

  def test_r_human_name(self):
    r = re.compile('\W' + r_human_name, re.MULTILINE)

    x = r.search('что-то Мироздания С.К., который был')
    self.assertEqual('Мироздания С.К.', x['human_name'])

    x = r.search('что-то Мироздания С. К., который был')
    self.assertEqual('Мироздания С. К.', x['human_name'])

    x = r.search('что-то Мироздания С. , который был')
    self.assertEqual('Мироздания С. ', x['human_name'])

    x = r.search('что-то  С. , который был')
    self.assertEqual(None, x)

    x = r.search('что-то друГое  С. R. , который был')
    self.assertEqual(None, x)

    x = r.search('что-то Мироздания Леонидыч Крупица, Который "был" таков')
    self.assertEqual('Мироздания Леонидыч Крупица', x['human_name'])

    x = r.search('что-то Мироздания Крупица, который был')
    self.assertEqual('Мироздания Крупица', x['human_name'])

  def test_replace_ip(self):
    r = sub_ip_quoter

    replacer = r[1]
    pattern = r[0]

    xx = pattern.sub(replacer, 'ИП Петров В.К.')
    self.assertEqual('ИП «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'Индивидуальная предприниматель Петров В.К.')
    self.assertEqual('Индивидуальная предприниматель «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда ИП Петров В.К.')
    self.assertEqual('некогда ИП «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда Индивидуальная предприниматель Петров В.К.')
    self.assertEqual('некогда Индивидуальная предприниматель «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда ИП Петров В.К.')
    self.assertEqual('некогда ИП «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда Индивидуальная предприниматель Мироздания Леонидыч Крупица')
    self.assertEqual('некогда Индивидуальная предприниматель «Мироздания Леонидыч Крупица»', xx)

    xx = pattern.sub(replacer, 'некогда ПРИНЦИПАЛ Мироздания Леонидыч Крупица')
    self.assertEqual('некогда ПРИНЦИПАЛ Мироздания Леонидыч Крупица', xx)

    t = """директора Регион Сибирь, и Индивидуальный предприниматель Петров В.В., именуемый в дальнейшем «Исполнитель»"""
    xx = pattern.sub(replacer, t)
    self.assertEqual(
      'директора Регион Сибирь, и Индивидуальный предприниматель «Петров В.В.», именуемый в дальнейшем «Исполнитель»',
      xx)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
