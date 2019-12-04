import datetime
import re

from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag

'''
refer https://github.com/nemoware/document-parser/blob/24013f562a8bc853134e116531f06ab9edcc0b00/src/main/java/com/nemo/document/parser/DocumentParser.java#L29
'''

date_regex_str = r"(?P<day>[1-2][0-9]|3[01]|0?[1-9]).\s*(?P<month>1[0-2]|0[1-9]|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря).\s*(?P<year>[1-2]\d{3})"
date_regex_c = re.compile(date_regex_str)

months_short = ["янв", "фев", "мар", "апр", "ма", "июн",
                "июл", "авг", "сен", "окт", "ноя", "дек"]

document_number_c = re.compile(r"[№N][ \t]*(?P<number>\S+)(\s+|$)")
document_number_valid_c = re.compile(r"([A-Za-zА-Яа-я0-9]+)")


def find_document_date(doc: LegalDocument, tagname='date') -> SemanticTag or None:
  head: LegalDocument = doc[0:HyperParameters.protocol_caption_max_size_words]

  try:
    findings = re.finditer(date_regex_c, head.text)
    if findings:
      finding = next(findings)
      _date = parse_date(finding)
      span = head.tokens_map.token_indices_by_char_range(finding.span())
      return SemanticTag(tagname, _date, span)
  except:
    pass
  return None


def find_document_number(doc: LegalDocument, tagname='number') -> SemanticTag or None:
  head: LegalDocument = doc[0:HyperParameters.protocol_caption_max_size_words]

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


def parse_date(finding):
  month = _get_month_number(finding["month"])
  year = int(finding['year'])
  day = int(finding['day'])

  _date = datetime.datetime(year, month, day)
  return _date


def _get_month_number(m):
  i = months_short.index(m[0:3])
  return i + 1


if __name__ == '__main__':
  doc = LegalDocument(
    'Договор пожертвования N 16-89/44 г. Санкт-Петербург                     «11» декабря 2018 год.\nМуниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем «Благополучатель»')
  doc.parse()
  tag = find_document_number(doc)
  print(tag)
  tag = find_document_date(doc)
  print(tag)
