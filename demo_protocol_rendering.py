from legal_docs import LegalDocument
from renderer import *


class ProtocolRenderer(AbstractRenderer):

  # def winning_patterns_to_html(self, _tokens, ranges, winning_patterns, _range,
  #                              colormaps=['Reds', 'Purples', 'Blues', 'Greens', 'Greys']):
  #   vmin = -ranges[1]
  #   vmax = -ranges[0]
  #
  #   #     print("winning_patterns_to_html _range", _range, "min max=", ranges)
  #
  #   norm = mpl.colors.Normalize(vmax=vmax, vmin=vmin)
  #
  #   cmaps = []
  #
  #   #     print (colormaps)
  #   for n in colormaps:
  #     cmap = mpl.cm.get_cmap(n)
  #     cmaps.append(cmap)
  #
  #   html = ""
  #
  #   for d in _range:
  #     winning_pattern_i = winning_patterns[d][0]
  #     colormap = cmaps[winning_pattern_i % len(colormaps)]
  #     normed = norm(-winning_patterns[d][1])
  #     color = mpl.colors.to_hex(colormap(normed))
  #     html += '<span title="' + '{} {:.2f}'.format(d, winning_patterns[d][
  #       1]) + '" style="background-color:' + color + '">' + str(
  #       _tokens[d]) + " </span>"
  #     if _tokens[d] == '\n':
  #       html += "<br>"
  #
  #   return html
  #
  # def _render_doc_subject_fragments(self, doc):
  #   #     print(doc.per_subject_distances)
  #
  #   _html = ""
  #   if doc.per_subject_distances is not None:
  #
  #     type = "Договор  благотворительного пожертвования"
  #     if doc.per_subject_distances[0] > doc.per_subject_distances[1]:
  #       type = "Договор возмездного оказания услуг"
  #
  #     _html += "<h3>" + type + "</h3>"
  #
  #     colormaps = ['PuRd'] * 5 + ['Blues'] * 7 + ['Greys']
  #
  #     _html += as_headline_4('Предмет договора')
  #
  #     for region in [doc.subj_range]:
  #       _html += self.winning_patterns_to_html(_tokens=doc.tokens, ranges=doc.subj_ranges,
  #                                         winning_patterns=doc.winning_subj_patterns, _range=region,
  #                                         colormaps=colormaps)
  #
  #   return _html
  #
  # def render_subject(self, counter):
  #   html = as_headline_3('Предмет документа (X):')   + self.subject_type_weights_to_html(counter)
  #   display(HTML(html))
  #
  # def print_results(self, _doc: LegalDocument, results=None):
  #
  #   if results is None:
  #     results = _doc.found_sum
  #
  #   result, (start, end), sentence, meta = results
  #
  #   html = "<hr>"
  #
  #   html += self._render_doc_subject_fragments(_doc)
  #
  #   if result is None:
  #     html += '<h2 style="color:red">СУММА НЕ НАЙДЕНА</h2>'
  #   else:
  #     html += '<h2>' + str(result[0]) + ' ' + str(result[1]) + '</h2>'
  #
  #   for key in meta.keys():
  #     html += '<div style="font-size:9px">' + str(key) + " = " + str(meta[key]) + "</div>"
  #
  #   display(HTML(html))
  #   self.render_color_text(_doc.tokens[start:end], _doc.sums[start:end])
  #
  # def subject_type_weights_to_html(self, counter):
  #   dict = {
  #     't_dea': 'Сделка',
  #     't_cha': 'Благотворительность',
  #     't_org': 'Организационные решения'
  #   }
  #
  #   maxkey = "None"
  #   for key in dict:
  #     if counter[key] > counter[maxkey]:
  #       maxkey = key
  #
  #   html = ""
  #   for key in dict:
  #     templ = "<div>{}: {}</div>"
  #     if key == maxkey:
  #       templ = '<b style="font-size:135%; color:maroon">{}: {}</b>'
  #     html += templ.format(counter[key], dict[key])
  #
  #   return html
