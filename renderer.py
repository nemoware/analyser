class AbstractRenderer:
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
