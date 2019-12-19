import re

from analyser.doc_dates import get_doc_head
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag

document_number_c = re.compile(r"[№N#]\s*(?P<number>[A-Za-zА-Яа-я0-9]{1,4}([ \/\-][0-9]{1,4}){0,3})")
document_number_valid_c = re.compile(r"([A-Za-zА-Яа-я0-9]+)")


def find_document_number(doc: LegalDocument, tagname='number') -> SemanticTag or None:
  head: LegalDocument = get_doc_head(doc)

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


if __name__ == '__main__':
  doc = LegalDocument(
    'Договор пожертвования N 16-89/44 г. Санкт-Петербург                     «11» декабря 2018 год.\nМуниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем «Благополучатель»')
  doc.parse()
  tag = find_document_number(doc)
  print(tag)
