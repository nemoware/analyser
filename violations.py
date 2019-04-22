from ml_tools import ProbableValue, np
from renderer import as_warning, as_offset, as_error_html, as_msg, as_quote, as_currency
from text_tools import untokenize
from transaction_values import ValueConstraint


class ViolationsFinder:


  def find_ranges_by_group(self, charter_constraints, m_convert, verbose=False):
    ranges_by_group = {}
    for head_group in charter_constraints:
      #     print('-' * 20)
      group_c = charter_constraints[head_group]
      data = self._combine_constraints_in_group(group_c, m_convert, verbose)
      ranges_by_group[head_group] = data
    return ranges_by_group

  @staticmethod
  def _combine_constraints_in_group(group_c, m_convert, verbose=False):
    # print(group_c)
    # print(group_c['section'])

    data = {
      'name': group_c['section'],
      'ranges': {}
    }

    sentences = group_c['sentences']
    #   print (charter_constraints[head_group]['sentences'])
    sentence_id = 0
    for sentence in sentences:
      constraint_low = None
      constraint_up = None

      sentence_id += 1
      #     print (sentence['constraints'])

      s_constraints = sentence['constraints']
      # большие ищем
      maximals = [x for x in s_constraints if x.value.sign > 0]

      if len(maximals) > 0:
        constraint_low = min(maximals, key=lambda item: m_convert(item.value).value)
        # if verbose:
        #   print("all maximals:")
        #   self.renderer.render_values(maximals)
        #   print('\t\t\t constraint_low', constraint_low.value.value)
        #   self.renderer.render_values([constraint_low])

      minimals = [x for x in s_constraints if x.value.sign <= 0]
      if len(minimals) > 0:
        constraint_up = min(minimals, key=lambda item: m_convert(item.value).value)
        # if verbose:
        #   print("all: minimals")
        #   self.renderer.render_values(minimals)
        #   print('\t\t\t constraint_upper', constraint_up.value.value)
        #   self.renderer.render_values([constraint_up])
        #   print("----X")

      if constraint_low is not None or constraint_up is not None:
        data['ranges'][sentence_id] = VConstraint(constraint_low, constraint_up, group_c)

    return data
    # ==================================================================VIOLATIONS

class VConstraint:
  def __init__(self, lower, upper, head_group):
    self.lower = ProbableValue(ValueConstraint(0, 'RUB', +1, context=[["-"]]), 0)
    self.upper = ProbableValue(ValueConstraint(np.inf, 'RUB', -1, context=[["-"]]), 0)

    if lower is not None:
      self.lower = lower

    if upper is not None:
      self.upper = upper

    self.head_group = head_group

  @staticmethod
  def maybe_convert(v: ValueConstraint, convet_m):
    html = ""
    v_converted = v
    if v.currency != 'RUB':
      v_converted = convet_m(v)
      html += as_warning(f"конвертация валют {as_currency(v)} --> RUB ")
      html += as_offset(as_warning(f"примерно: {as_currency(v)} ~~  {as_currency(v_converted)}  "))
    return v, v_converted, html

  def check_contract_value(self, _v: ProbableValue, convet_m, renderer):
    greather_lower = False
    greather_upper = False

    if _v is None:
      return as_error_html("сумма контракта неизвестна")
    v: ValueConstraint = _v.value

    if v is None:
      return as_error_html("сумма контракта не верна")

    if v.value is None:
      return as_error_html(f"сумма контракта не верна {v.currency}")
    ###----

    lower_v = None
    upper_v = None
    if self.lower is not None:
      lower_v: ValueConstraint = self.lower.value
    if self.upper is not None:
      upper_v: ValueConstraint = self.upper.value

    html = as_msg(f"диапазон: {as_currency(lower_v)} < ..... < {as_currency(upper_v)}")

    v, v_converted, h = self.maybe_convert(v, convet_m)
    html += h

    if self.lower is not None:
      lower_v: ValueConstraint = self.lower.value
      lower_v, lower_converted, h = self.maybe_convert(lower_v, convet_m)
      html += h

      if v_converted.value >= lower_converted.value:
        greather_lower = True
        html += as_warning("требуется одобрение...".upper())
        html += as_warning(
          f"сумма договора  {as_currency(v_converted)}  БОЛЬШЕ нижней пороговой {as_currency(lower_converted)} ")
        html += as_quote(untokenize(lower_v.context[0]))

    if self.upper is not None:

      upper_v: ValueConstraint = self.upper.value
      upper_v, upper_converted, h = self.maybe_convert(upper_v, convet_m)
      html += h

      if v_converted.value >= upper_converted.value:

        html += as_error_html(
          f"сумма договора  {as_currency(v_converted)} БОЛЬШЕ верхней пороговой {as_currency(upper_converted)} ")

      elif greather_lower:
        head_name = self.head_group['section']
        html += as_error_html(f'требуется одобрение со стороны "{head_name.upper()}"')

        if lower_v.context is not None:
          html += as_quote(renderer.to_color_text(lower_v.context[0], lower_v.context[1], _range=[0, 1]))
        if upper_v.context is not None:
          html += '<br>'
          html += as_quote(renderer.to_color_text(upper_v.context[0], upper_v.context[1], _range=[0, 1]))

    return html
