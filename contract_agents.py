import re


def ru_cap(xx):
  return '\s+'.join([f'[{x[0].upper()}{x[0].lower()}]{x[1:-2]}[а-я]{{0,3}}' for x in xx.split(' ')])


q_re_open = '\'\"«'
q_re_close = '\'\"»'

sf = '[а-я]{1,3}'

r_few_words = r'\s+[А-Я]{1}[а-я\-, ]{1,80}\s+'
r_few_words_s = r'\s+[А-Яа-я\-, ]{1,80}\s+'

ORG_TYPES_re = [
  ru_cap('Акционерное общество'), 'АО',
  ru_cap('Закрытое акционерное общество'), 'ЗАО',
  ru_cap('Открытое акционерное общество'), 'ОАО',
  ru_cap('Государственное автономное учреждение') + r_few_words_s,
  ru_cap('Муниципальное бюджетное учреждение'),
  ru_cap('Общественная организация'),
  ru_cap('Общество с ограниченной ответственностью'), 'ООО',
  ru_cap('Некоммерческая организация'),
  ru_cap('организация'),
  ru_cap('Благотворительный фонд'),

  ru_cap('Фонд') + r_few_words_s,
  r_few_words_s + ru_cap('учреждение'),
  ru_cap('Индивидуальный предприниматель'), 'ИП'

]


r_types_ = '|'.join([x for x in ORG_TYPES_re])
r_types = f'({r_types_})'

r_name_a = r'([–А-Яa-я\- ]{0,30})'
r_name = r'([–А-Яa-я\- ]{0,50})'

r_quote_open = r'([«"<]|[\']{2})'
r_quote_close = r'([»">]|[\']{2})'

r_alter = r'(\s+\(.{1,40}\))?'

complete_re = re.compile(r_types + r'\s*' + r_name_a + r_quote_open + r_name + r_quote_close + r_alter,
                         re.MULTILINE)


def find_org_names(txt):
  def clean(x):
    if x is None: return x
    return x.replace('\t', ' ').replace('\n', ' ').replace(' – ', '-')

  def to_dict(_org):
    return {
      '1_type': clean(_org[1]),
      '2_type_ext': clean(_org[2]),
      '3_name': f'"{clean(_org[4])}"',
      '4_alt_name': clean(_org[6])
    }

  def find_from(start):
    _next = txt[start:]
    r = complete_re.search(_next)
    if r is not None:
      return to_dict(r), start + r.span()[1]
    else:
      return None, None

  org_names = {}
  r, end = find_from(0)
  if r is not None:
    org_names[0] = r

    r2, end = find_from(end)
    while r2 is not None and r2['3_name'].lower() == r['3_name'].lower():
      r2, end = find_from(end)

    if r2 is not None:
      org_names[1] = r2

  return org_names
