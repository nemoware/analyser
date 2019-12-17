import datetime
import re

from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag

'''
refer https://github.com/nemoware/document-parser/blob/24013f562a8bc853134e116531f06ab9edcc0b00/src/main/java/com/nemo/document/parser/DocumentParser.java#L29
'''

months_short_temp = [r"янв", r"фев", r"мар", r"апр", r"ма[йя]", r"июн",
                     r"июл", r"авг", r"сен", r"окт", r"ноя", r"дек"]
months_short = [re.compile(c, re.UNICODE | re.IGNORECASE) for c in months_short_temp]
_months_short_combined = '|'.join(months_short_temp)
_month_no = r'1[0-2]|0[1-9]'
_months_long_combined = 'января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря'
_date_month_ = f'({_months_short_combined})|({_months_long_combined})|({_month_no})'

_date_day = r'«?(?P<day>[1-2][0-9]|3[01]|0?[1-9])»?'
_date_year = r'(?P<year>[1-2]\d{3})'
_date_month = f'(?P<month>{_date_month_})'
_date_separator = r'(\s*|\-|\.)'
# date_regex_str = r"(?P<day>[1-2][0-9]|3[01]|0?[1-9]).\s*(?P<month>1[0-2]|0[1-9]|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря).\s*(?P<year>[1-2]\d{3})"
date_regex_str = f'{_date_day}{_date_separator}{_date_month}{_date_separator}{_date_year}'
date_regex_c = re.compile(date_regex_str, re.IGNORECASE | re.UNICODE)

document_number_c = re.compile(r"[№N][ \t]*(?P<number>\S+)(\s+|$)")
document_number_valid_c = re.compile(r"([A-Za-zА-Яа-я0-9]+)")


def find_document_date(doc: LegalDocument, tagname='date') -> SemanticTag or None:
  head: LegalDocument = doc[0:HyperParameters.protocol_caption_max_size_words]
  c_span, _date = find_date(head.text)
  if c_span is None:
    return None
  span = head.tokens_map.token_indices_by_char_range(c_span)
  return SemanticTag(tagname, _date, span)


def find_date(text: str) -> ([], datetime.datetime):
  try:
    findings = re.finditer(date_regex_c, text)
    if findings:
      finding = next(findings)
      _date = parse_date(finding)
      if _date:
        return finding.span(), _date
  except:
    pass

  return None, None


def find_document_number(doc: LegalDocument, tagname='number') -> SemanticTag or None:
  head: LegalDocument = doc[0:HyperParameters.protocol_caption_max_size_words]
  # TODO: take first paragraph. If it is short, take head.

  try:
    findings = re.finditer(document_number_c, head.text)
    if findings:
      finding = next(findings)
      _number = finding['number']
      if document_number_valid_c.match(_number):
        span = head.tokens_map.token_indices_by_char_range(finding.span())
        return SemanticTag(tagname, _number, span)
  except:
    pass
  return None


def find_document_number_in_subdoc(doc: LegalDocument, tagname='number', parent=None) -> [SemanticTag]:
  ret = []
  findings = re.finditer(document_number_c, doc.text)
  if findings:
    for finding in findings:
      _number = finding['number']
      if document_number_valid_c.match(_number):
        span = doc.tokens_map.token_indices_by_char_range(finding.span())
        tag = SemanticTag(tagname, _number, span, parent=parent)
        tag.offset(doc.start)
        ret.append(tag)
      else:
        print('invalid', _number)

  return ret


def parse_date(finding) -> ([], datetime.datetime):
  month = _get_month_number(finding["month"])
  year = int(finding['year'])
  day = int(finding['day'])

  if month > 0:
    _date = datetime.datetime(year, month, day)
    return _date


def _get_month_number(m):
  if m.isdigit():
    try:
      return int(m)
    except:
      pass

  for p in range(len(months_short)):
    if re.match(months_short[p], m):
      return p + 1
  return -1


if __name__ == '__main__':
  doc = LegalDocument(
    'Договор пожертвования N 16-89/44 г. Санкт-Петербург                     «11» декабря 2018 год.\nМуниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем «Благополучатель»')
  doc.parse()
  tag = find_document_number(doc)
  print(tag)
  tag = find_document_date(doc)
  print(tag)
