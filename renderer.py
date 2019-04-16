from transaction_values import ValueConstraint


class AbstractRenderer:

  def sign_to_text(self, sign: int):
    if sign < 0: return " &lt; "
    if sign > 0: return " &gt; "
    return ' = '

  def value_to_html(self, vc: ValueConstraint):
    color = '#333333'
    if vc.sign > 0:
      color = '#993300'
    elif vc.sign < 0:
      color = '#009933'

    return f'<b style="color:{color}">{sign_to_text(vc.sign)} {vc.currency} {vc.value:20,.2f}</b> '

  def render_value_section_details(self, value_section_info):
    raise NotImplementedError()

  def to_color_text(self, tokens, weights, colormap='coolwarm', print_debug=False, _range=None):
    raise NotImplementedError()

  def render_color_text(self, tokens, weights, colormap='coolwarm', print_debug=False, _range=None):
    raise NotImplementedError()

  def print_results(self, doc, results):
    raise NotImplementedError()

  def render_values(self, values):
    raise NotImplementedError()
