from typing import List

from ml_tools import ProbableValue
from transaction_values import ValueConstraint

head_types_colors = {'head.directors': 'crimson',
                     'head.all': 'orange',
                     'head.gen': 'blue',
                     'head.shareholders': '#666600',
                     'head.pravlenie': '#0099cc',
                     'head.unknown': '#999999'}


def as_smaller(x):
  return f'<span style="font-size:80%;">{x}</span>'

def as_error_html(txt):
  return f'<div style="color:red">⚠️ {txt}</div>'


def as_warning(txt):
  return f'<div style="color:orange">⚠️ {txt}</div>'


def as_msg(txt):
  return f'<div>{txt}</div>'


def as_quote(txt):
  return f'<i style="margin-top:0.2em; margin-left:2em; font-size:90%">"...{txt} ..."</i>'


def as_headline_2(txt):
  return f'<h2>{txt}</h2>'


def as_headline_3(txt):
  return f'<h3 style="margin:0">{txt}</h3>'

def as_headline_4(txt):
  return f'<h4 style="margin:0">{txt}</h4>'

def as_offset(txt):
  return f'<div style="margin-left:2em">{txt}</div>'


def as_currency(v):
  if v is None: return "any"
  return f'{v.value:20,.0f} {v.currency} '


def as_error_html(txt):
  return f'<div style="color:red">{txt}</div>'


def as_warning(txt):
  return f'<div style="color:orange">{txt}</div>'


def as_msg(txt):
  return f'<div>{txt}</div>'


def as_quote(txt):
  return f'<i style="margin-top:0.2em; margin-left:2em; font-size:90%">{txt}</i>'


def as_headline_2(txt):
  return f'<h2>{txt}</h2>'


def as_headline_3(txt):
  return f'<h3 style="margin:0">{txt}</h3>'


class AbstractRenderer:

  def sign_to_text(self, sign: int):
    if sign < 0: return " < "
    if sign > 0: return " > "
    return ' = '

  def sign_to_html(self, sign: int):
    if sign < 0: return " &lt; "
    if sign > 0: return " &gt; "
    return ' = '

  def value_to_html(self, vc: ValueConstraint):
    color = '#333333'
    if vc.sign > 0:
      color = '#993300'
    elif vc.sign < 0:
      color = '#009933'

    return f'<b style="color:{color}">{self.sign_to_html(vc.sign)} {vc.currency} {vc.value:20,.2f}</b> '

  def render_value_section_details(self, value_section_info):
    pass

  def to_color_text(self, tokens, weights, colormap='coolwarm', print_debug=False, _range=None):
    pass

  def render_color_text(self, tokens, weights, colormap='coolwarm', print_debug=False, _range=None):
    pass

  def print_results(self, doc, results):
    raise NotImplementedError()

  def render_values(self, values: List[ProbableValue]):
    for pv in values:
      vc = pv.value
      s = f'{self.sign_to_text(vc.sign)} \t {vc.currency} \t {vc.value:20,.2f} \t {pv.confidence:20,.2f} '
      print(s)

  def render_contents(self, doc):
    pass


class SilentRenderer(AbstractRenderer):
  pass


import numpy as np

''' AZ:- 🌈 -----🌈 ------🌈 --------------------------END-Rendering COLORS--------'''

def mixclr(color_map, dictionary, min_color=None, _slice=None):
  reds = None
  greens = None
  blues = None

  for c in dictionary:
    vector = np.array(dictionary[c])
    if _slice is not None:
      vector = vector[_slice]

    if reds is None:
      reds = np.zeros(len(vector))
    if greens is None:
      greens = np.zeros(len(vector))
    if blues is None:
      blues = np.zeros(len(vector))

    vector_color = color_map[c]

    reds += vector * vector_color[0]
    greens += vector * vector_color[1]
    blues += vector * vector_color[2]


  if min_color is not None:
    reds += min_color[0]
    greens += min_color[1]
    blues += min_color[2]

  def cut_(x):
    up = [min(i, 1) for i in x]
    down = [max(i, 0) for i in up]
    return down

  return np.array([cut_(reds), cut_(greens), cut_(blues)]).T



def to_multicolor_text(tokens, vectors, colormap, min_color=None, _slice=None) -> str:
  if _slice is not None:
    tokens = tokens[_slice]

  colors = mixclr(colormap, vectors, min_color=min_color, _slice=_slice)
  html = ''
  for i in range(len(tokens)):
    c = colors[i]
    r = int(255 * c[0])
    g = int(255 * c[1])
    b = int(255 * c[2])
    if tokens[i] == '\n':
      html += '<br>'
    html += f'<span style="background:rgb({r},{g},{b})"> {tokens[i]} </span>'
  return html

''' AZ:- 🌈 -----🌈 ------🌈 --------------------------END-Rendering COLORS--------'''