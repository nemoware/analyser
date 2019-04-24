# -*- coding: utf-8 -*-
"""DEMO V Charters: Value & Org Extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J5e1NpjH8bmA7hEYKJG-RLt1H9e1M5N3

## Config
"""

#@title ## Settings { run: "auto", vertical-output: true, form-width: "750px", display-mode: "form" }

hyperparameters={}


#@markdown ## - DEMO
upload_enabled = True  #@param {type: "boolean"}

#@markdown ## - Batch
#@markdown - пакетная обработка нескольких уставов из `GoogleDrive/GazpromOil/Charters`   
#@markdown - запись результатов в https://docs.google.com/spreadsheets/d/13Clx3Rzd3BWC2E2b-GzkSLuAeuwNmIYssgir-CCvYHc

run_batch_processing = True  #@param {type: "boolean"}
read_docs_from_google_drive = True  #@param {type: "boolean"}


#@markdown ## - dev
dev_mode = False  #@param {type: "boolean"}
#@markdown - запуск тестов после декларации методов и функций
perform_test_on_small_doc = False  #@param {type: "boolean"}
git_branch = "structured_2" #@param {type:"string"}

#@markdown ## - Embedding module
embeddings_layer='elmo'  #@param ["elmo", "word_emb" ]
database = "news" #@param ["wiki", "twitter", "news"]


hyperparameters['embeddings.layer']=embeddings_layer
hyperparameters ['database']=database
print()

print(str(hyperparameters))

"""## Init"""

import tensorflow as tf
import tensorflow_hub as hub

print(tf.__version__)
elmo = None

_database = hyperparameters ['database']
print('_database = ',_database)



_databases={
    #Twitter
    'twitter':'https://storage.googleapis.com/az-nlp/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz',
                         
    
    #Russian WMT News
    'news':'https://storage.googleapis.com/az-nlp/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz',
            
    #Wikipedia
    'wiki':'http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz'
}


elmo = hub.Module(_databases[_database], trainable=False) #twitter

"""### Import from GitHub"""

!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/text_tools.py
!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/embedding_tools.py
!rm ml_tools.py  
!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/ml_tools.py
!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/text_normalize.py  
!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/patterns.py 
!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/transaction_values.py  

from transaction_values import *
from embedding_tools import *

# from split import *

!rm doc_structure.py  
!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/doc_structure.py 
!rm legal_docs.py  
!wget https://raw.githubusercontent.com/compartia/nlp_tools/structured_2/legal_docs.py  
from legal_docs import *
from doc_structure import *

"""# Code (common)"""

REPORTED_MOVED={}

def at_github(fn):
  @wraps(fn)
  @wraps(fn)
  def with_reporting(*args, **kwargs):
    if fn.__name__ not in REPORTED_MOVED:
      REPORTED_MOVED[fn.__name__] = 1
      print("----WARNING!: function {} must be imported from github".format(fn.__name__) )
 
    ret = fn(*args, **kwargs)
    return ret

  return with_reporting

"""## FS utils"""


!pip install docx2txt
!sudo apt-get install antiword
  

import docx2txt, sys, os
from google.colab import files



def read_doc(fn):
  
  text = ''
  try:
    text = docx2txt.process(fn)
  except:
    print("Unexpected error:", sys.exc_info())
    os.system('antiword -w 0 "' + fn + '" > "' + fn + '.txt"')
    with open(fn + '.txt') as f:
      text = f.read()

  return text


#-------
def interactive_upload():
  print('select .docx files:')
  uploaded = files.upload()
  docs=[]
  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))

    with open(fn, "wb") as df:
      df.write(uploaded[fn])
      df.close()

    # extract text
  
    text = ''
    try:
      text = docx2txt.process(fn)
    except:
      print("Unexpected error:", sys.exc_info())
      os.system('antiword -w 0 "' + fn + '" > "' + fn + '.txt"')
      with open(fn + '.txt') as f:
        text = f.read()
      #os.remove(fn+'.txt') #fn.txt was just to read, so deleting  
  
#     print(text)
    docs.append(text)
    return docs

"""## Embedder"""

class ElmoEmbedder(AbstractEmbedder):

  def __init__(self, elmo):
    self.elmo = elmo
    self.config = tf.ConfigProto()
    self.config.gpu_options.allow_growth=True

  def embedd_tokenized_text(self, words, lens):
    with tf.Session(config=self.config) as sess:
      embeddings = self.elmo(
        inputs={
          "tokens": words,
          "sequence_len": lens
        },
        signature="tokens",
        as_dict=True)[ hyperparameters['embeddings.layer'] ]

      sess.run(tf.global_variables_initializer())
      out = sess.run(embeddings)
#       sess.close()


    return out, words

  def get_embedding_tensor(self, str, type=hyperparameters['embeddings.layer'], signature="default"):
    embedding_tensor = self.elmo(str, signature=signature, as_dict=True)[type]

    with tf.Session(config=self.config) as sess:
      sess.run(tf.global_variables_initializer())
      embedding = sess.run(embedding_tensor)
#       sess.close()

    return embedding


embedder = ElmoEmbedder(elmo)



"""## Rendering"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

def render_org(org):
  txt_html = to_color_text(org['tokens'], org['attention_vector'], _range=[0,1])
  html = '<div style="background:#eeeeff; padding:0.5em"> recognized NE(s): <br><br> org type:<h3 style="margin:0">  {} </h3>org full name:<h2 style="margin:0">  {} </h2> <br>quote: <div style="font-size:90%; background:white">{}</div> </div>'.format( org['type_name'],org['name'], txt_html )
  display(HTML(html))
  
  
def to_color_text(tokens, weights, colormap='coolwarm', print_debug=False, _range=None):
  #   weights = _weights *-1
  if len(tokens)==0:
#     raise ValueError("don't know how to render emptiness")
    return " - empty -"
  if len(weights) != len(tokens):
    raise ValueError("number of weights differs weights={} tokens={}".format(len(weights), len(tokens)))

  #   if()
  vmin = weights.min()
  vmax = weights.max()

  if _range is not None:
    vmin = _range[0]
    vmax = _range[1]

  if print_debug:
    print(vmin, vmax)

  norm = mpl.colors.Normalize(vmin=vmin - 0.5, vmax=vmax)
  html = ""
  cmap = mpl.cm.get_cmap(colormap)

  for d in range(0, len(weights)):
    word = tokens[d]
    if word == ' ':
      word = '&nbsp;_ '
    
    html += '<span title="{} {:.4f}" style="background-color:{}">{} </span>'.format(
      d,
      weights[d],
      mpl.colors.to_hex(cmap(norm(weights[d]))),
      word)

    #     html+='<span style="background-color:' +mpl.colors.to_hex(cmap(norm(weights[d]) ))+ '">' + str(tokens[d]) + " </span>"
    if tokens[d] == '\n':
      html += "<br>"

  return html


def render_color_text(tokens, weights, colormap='coolwarm', print_debug=False, _range=None):
  html = to_color_text(tokens, weights, colormap, print_debug, _range)
  display(HTML(html))


def winning_patterns_to_html(_tokens, ranges, winning_patterns, _range,
                             colormaps=['Reds', 'Purples', 'Blues', 'Greens', 'Greys']):
  vmin = -ranges[1]
  vmax = -ranges[0]

  #     print("winning_patterns_to_html _range", _range, "min max=", ranges)

  norm = mpl.colors.Normalize(vmax=vmax, vmin=vmin)

  cmaps = []

  #     print (colormaps)
  for n in colormaps:
    cmap = mpl.cm.get_cmap(n)
    cmaps.append(cmap)

  html = ""

  for d in _range:
    winning_pattern_i = winning_patterns[d][0]
    colormap = cmaps[winning_pattern_i % len(colormaps)]
    normed = norm(-winning_patterns[d][1])
    color = mpl.colors.to_hex(colormap(normed))
    html += '<span title="' + '{} {:.2f}'.format(d, winning_patterns[d][
      1]) + '" style="background-color:' + color + '">' + str(
      _tokens[d]) + " </span>"
    if _tokens[d] == '\n':
      html += "<br>"

  return html


def _render_doc_subject_fragments(doc):
  #     print(doc.per_subject_distances)

  _html = ""
  if doc.per_subject_distances is not None:

    type = "Договор  благотворительного пожертвования"
    if doc.per_subject_distances[0] > doc.per_subject_distances[1]:
      type = "Договор возмездного оказания услуг"

    _html += "<h3>" + type + "</h3>"

    colormaps = ['PuRd'] * 5 + ['Blues'] * 7 + ['Greys']

    _html += "<h4> Предмет договора:</h4>"

    for region in [doc.subj_range]:
      _html += winning_patterns_to_html(_tokens=doc.tokens, ranges=doc.subj_ranges,
                                        winning_patterns=doc.winning_subj_patterns, _range=region,
                                        colormaps=colormaps)

  return _html


currencly_map = {
  'доллар': 'USD',
  'евро': 'EUR',
  'руб': 'RUR'
}


def sum_to_html(result, prefix=''):
  html = ""
  if result is None:
    html += '<h3 style="color:red">СУММА НЕ НАЙДЕНА</h3>'
  else:
    html += '<h3>{}{} {:20,.2f}</h3>'.format(prefix, currencly_map[result[1]], result[0])
  return html


def print_results(_doc, results=None):
  if results is None:
    results = _doc.found_sum

  result, (start, end), sentence, meta = results

  html = "<hr>"

  html += _render_doc_subject_fragments(_doc)

  html += sum_to_html(result)

  for key in meta.keys():
    html += '<div style="font-size:9px">' + str(key) + " = " + str(meta[key]) + "</div>"

  display(HTML(html))
  render_color_text(_doc.tokens[start:end], _doc.sums[start:end])


def render_sections(doc, weights):
  if weights is None:
    weights = np.zeros(len(doc.tokens))
    for subdoc in doc.subdocs:
      weights[subdoc.start] = 1

  fig = plt.figure(figsize=(20, 6))
  ax = plt.axes()
  ax.plot(weights, alpha=0.5, color='green', label='Sections');
  plt.title('Sections')

  for subdoc in doc.subdocs:
    print(subdoc.filename, '-' * 20)
    render_color_text(subdoc.tokens, weights[subdoc.start:subdoc.end], _range=[0, 1])

"""## Legal Doc Classes

### Document Structure
"""

# ------------------------------
def subdoc_between_lines(line_a: int, line_b: int, doc):
  _str = doc.structure.structure
  start = _str[line_a].span[1]
  if line_b is not None:
    end = _str[line_b].span[0]
  else:
    end = len(doc.tokens)
  return doc.subdoc(start, end)


# ------------------------------
@deprecated
def find_best_headline_by_pattern_prefix(headline_indices, embedded_headlines, pat_refix, threshold, render=False):
  distance_by_headline = []
  if render:
    print('headlines', '-' * 20)

  if render:
    fig = plt.figure(figsize=(20, 7))
    ax = plt.axes()

  off_ = 0

  attention_vs = []
  for headline_index, subdoc in zip(headline_indices, embedded_headlines):
    names, _c = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, pat_refix, relu_th=0.6)
    names = smooth_safe(names, 4)
    _max_id = np.argmax(names)
    _max = np.max(names)
    _sum = math.log(1 + np.sum(names[_max_id - 1:_max_id + 2]))
    distance_by_headline.append(_max + _sum)
    attention_vs.append(names)

    if render:
      ax.plot(names + off_, alpha=0.5, label=str(headline_index));
      print('dist={} sum={} d+s={}'.format(_max, _sum, _max + _sum))
      render_color_text(subdoc.tokens_cc, names, _range=[0, 2])
    off_ += 1

  if render:
    plt.legend(loc='upper left')
    plt.title("find_best_headline_by_pattern_prefix:" + pat_refix + " in headlines")

  bi = np.argmax(distance_by_headline)
  if distance_by_headline[bi] < threshold:
    raise ValueError('Cannot find headline matching pattern "{}"'.format(pat_refix))

  return bi, distance_by_headline, attention_vs[bi]


# ------------------------------
import math
from typing import List

from legal_docs import rectifyed_sum_by_pattern_prefix, LegalDocument
from ml_tools import smooth_safe



def _find_best_headline_by_pattern_prefix(headline_indices, embedded_headlines, pat_refix, threshold, render=False):
  distance_by_headline = []

  attention_vs = []
  for headline_index, subdoc in zip(headline_indices, embedded_headlines):
    names, _c = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, pat_refix, relu_th=0.6)
    names = smooth_safe(names, 4)
    _max_id = np.argmax(names)
    _max = np.max(names)
    _sum = math.log(1 + np.sum(names[_max_id - 1:_max_id + 2]))
    distance_by_headline.append(_max + _sum)
    attention_vs.append(names)

  bi = np.argmax(distance_by_headline)
  if distance_by_headline[bi] < threshold:
    raise ValueError('Cannot find headline matching pattern "{}"'.format(pat_refix))

  return bi, distance_by_headline, attention_vs[bi]


def match_headline_types(head_types_list, headline_indexes, embedded_headlines: List[LegalDocument], pattern_prefix,
                         threshold):
  best_indexes = {}
  for head_type in head_types_list:
    try:
      bi, distance_by_headline, attention_v = \
        _find_best_headline_by_pattern_prefix(headline_indexes, embedded_headlines, pattern_prefix + head_type,
                                              threshold,
                                              render=False)

      obj = {'headline.index': bi,
             'headline.type': head_type,
             'headline.confidence': distance_by_headline[bi],
             'headline.subdoc': embedded_headlines[bi],
             'headline.attention_v': attention_v}

      if bi in best_indexes:
        e_obj = best_indexes[bi]
        if e_obj['headline.confidence'] < obj['headline.confidence']:
          best_indexes[bi] = obj
      else:
        best_indexes[bi] = obj

    except Exception as e:
      print(e)
      pass

  return best_indexes


# ------------------------------
@deprecated
def map_headline_index_to_headline_type(headline_indexes, embedded_headlines, threshold):
  best_indexes = {}
  for head_type in head_types:
    try:
      bi, distance_by_headline, attention_v = \
        find_best_headline_by_pattern_prefix(headline_indexes, embedded_headlines, 'd_head_' + head_type, threshold,
                                             render=False)

      obj = {'headline.index': bi,
             'headline.type': head_type,
             'headline.confidence': distance_by_headline[bi],
             'headline.subdoc': embedded_headlines[bi],
             'headline.attention_v': attention_v}

      if bi in best_indexes:
        e_obj = best_indexes[bi]
        if e_obj['headline.confidence'] < obj['headline.confidence']:
          best_indexes[bi] = obj
      else:
        best_indexes[bi] = obj

    except Exception as e:
      print(e)
      pass

  return best_indexes



# ------------------------------
def _doc_section_under_headline(_doc, hl_struct, headline_indices, render=False):
  if render:
    print('_doc_section_under_headline:searching for section:', hl_struct['headline.type'])

  bi = hl_struct['headline.index']

  bi_next = bi + 1
  best_headline = headline_indices[bi]

  if bi_next < len(headline_indices):
    best_headline_next = headline_indices[bi_next]
  else:
    best_headline_next = None

  if render:
    print(
      '_doc_section_under_headline: best_headline:{} best_headline_next:{} bi:{}'.format(best_headline,
                                                                                         best_headline_next, bi),
      '_' * 40)

  subdoc = subdoc_between_lines(best_headline, best_headline_next, _doc)
  if len(subdoc.tokens) < 2:
    raise ValueError(
      'Empty "{}" section between headlines #{} and #{}'.format(hl_struct['headline.type'], best_headline, best_headline_next))

  # May be embedd
  if render:
    print('_doc_section_under_headline: embedding segment:', untokenize(subdoc.tokens_cc))

  

  return subdoc


# ------------------------------
def find_sections_by_headlines(best_indexes, _doc, headline_indexes, render=False):
  sections = {}

  for bi in best_indexes:

    """
    bi = {
        'headline.index': bi,
        'headline.type': head_type,
        'headline.confidence': distance_by_headline[bi],
        'headline.subdoc': embedded_headlines[bi],
        'headline.attention_v': attention_v}
    """
    hl = best_indexes[bi]
    
    if render:
      print('=' * 100)
      print(untokenize(hl['headline.subdoc'].tokens_cc))
      print('-' * 100)

    head_type = hl['headline.type']

    try:      
      hl['body.subdoc'] = _doc_section_under_headline(_doc, hl, headline_indexes, render=render)
      sections[head_type] = hl
      
    except ValueError as error:
      print(error)

  return sections

"""# Charter parsing-related code

### Constants
"""

# self.headlines = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie', 'name']

head_types = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie']

head_types_dict = {  'head.directors':'Совет директоров', 
                     'head.all':'Общее собрание участников/акционеров', 
                     'head.gen':'Генеральный директор', 
#                      'shareholders':'Общее собрание акционеров', 
                     'head.pravlenie':'Правление общества',
                     'head.unknown':'*Неизвестный орган управления*'}

head_types_colors = {  'head.directors':'crimson', 
                     'head.all':'orange', 
                     'head.gen':'blue', 
                     'head.shareholders':'#666600', 
                     'head.pravlenie':'#0099cc',
                     'head.unknown':'#999999'}


org_types={
    'org_unknown':'undefined', 
    'org_ao':'Акционерное общество', 
    'org_zao':'Закрытое акционерное общество', 
    'org_oao':'Открытое акционерное общество', 
    'org_ooo':'Общество с ограниченной ответственностью'}

"""## 1.  Patterns Factory 1

### HeadlinesPatternFactory
"""

class HeadlinesPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)
    self.patterns_dict = {}
    self._build_head_patterns()
    self.embedd()
    
    self.headlines = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie', 'name']

  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)
    
    head_prfx="статья 0"
            
    cp('headline.name.1', ('Полное', 'фирменное наименование', 'общества на русском языке:'))    
    cp('headline.name.2', ('', 'ОБЩИЕ ПОЛОЖЕНИЯ', ''))    
    cp('headline.name.3', ('', 'фирменное', ''))
    cp('headline.name.4', ('', 'русском', ''))
    cp('headline.name.5', ('', 'языке', ''))
    cp('headline.name.6', ('', 'полное', ''))
    
    
    cp('headline.head.all.1', (head_prfx, 'компетенции общего собрания акционеров', ''))
    cp('headline.head.all.1', (head_prfx, 'компетенции общего собрания участников', 'общества'))
    cp('headline.head.all.2', (head_prfx, 'собрание акционеров\n', ''))
    
    cp('headline.head.all.3', ('', 'компетенции', ''))
    cp('headline.head.all.4', ('', 'собрания', ''))
    cp('headline.head.all.5', ('', 'участников', ''))
    cp('headline.head.all.6', ('', 'акционеров', ''))
    
    
    cp('headline.head.directors.1', (head_prfx, 'компетенция совета директоров', 'общества'))
    cp('headline.head.directors.2', ('', 'совет директоров общества', ''))
    cp('headline.head.directors.3', ('', 'компетенции', ''))
    cp('headline.head.directors.4', ('', 'совета', ''))
    cp('headline.head.directors.5', ('', 'директоров', ''))
    
    
    cp('headline.head.pravlenie.1', (head_prfx, 'компетенции правления', ''))
    cp('headline.head.pravlenie.2', ('', 'компетенции', ''))
    cp('headline.head.pravlenie.3', ('', 'правления', ''))
#     cp('d_head_pravlenie.2', ('', 'общества', ''))
    
    cp('headline.head.gen.1', (head_prfx, 'компетенции генерального директора', ''))
    cp('headline.head.gen.2', ('', 'компетенции', ''))
    cp('headline.head.gen.3', ('', 'генерального', ''))
    cp('headline.head.gen.4', ('', 'директора', ''))
    

HPF = HeadlinesPatternFactory(embedder)

"""### PricePF

see (https://colab.research.google.com/drive/1w5KNrKn6O4GFM5dFEspeIVEF-mUwHGm_#scrollTo=1uz5CtBdETys&uniqifier=6)
"""

## DO NOT EDIT HERE
# see https://colab.research.google.com/drive/1w5KNrKn6O4GFM5dFEspeIVEF-mUwHGm_#scrollTo=1uz5CtBdETys&uniqifier=6
from patterns import AbstractPatternFactory, FuzzyPattern


class PriceFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self.patterns_dict = {}
 
    self._build_order_patterns()
    self._build_sum_margin_extraction_patterns()
    self._build_sum_patterns()
    self.embedd()

  def _build_order_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    prefix = 'принятие решения о согласии на совершение или о последующем одобрении'
    
    cp('d_order_4', (prefix, 'cделки', ', стоимость которой равна или превышает'))
    cp('d_order_5', (prefix, 'cделки', ', стоимость которой составляет менее'))
    
  def _build_sum_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)
    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'решений о совершении сделок '


    cp('sum_max1', (prefix + 'стоимость', 'не более 0', suffix))
    cp('sum_max2', (prefix + 'цена', 'не больше 0', suffix))
    cp('sum_max3', (prefix + 'стоимость <', '0', suffix))
    cp('sum_max4', (prefix + 'цена менее', '0', suffix))
    cp('sum_max5', (prefix + 'стоимость не может превышать', '0', suffix))  
    cp('sum_max6', (prefix + 'общая сумма может составить', '0', suffix))
    cp('sum_max7', (prefix + 'лимит соглашения', '0', suffix))
    cp('sum_max8', (prefix + 'верхний лимит стоимости', '0', suffix))
    cp('sum_max9', (prefix + 'максимальная сумма', '0', suffix))
    
     

  def _build_sum_margin_extraction_patterns(self):
    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'совершение сделок '

    # less than
    self.create_pattern('sum__lt_1', (prefix + 'стоимость', 'не более 0', suffix))
    self.create_pattern('sum__lt_2', (prefix + 'цена', 'не больше 0', suffix))
    self.create_pattern('sum__lt_3', (prefix + 'стоимость', '< 0', suffix))
    self.create_pattern('sum__lt_4', (prefix + 'цена', 'менее 0', suffix))
    self.create_pattern('sum__lt_4.1', (prefix + 'цена', 'ниже 0', suffix))
    self.create_pattern('sum__lt_5', (prefix + 'стоимость', 'не может превышать 0', suffix))
    self.create_pattern('sum__lt_6', (prefix + 'лимит соглашения', '0', suffix))
    self.create_pattern('sum__lt_7', (prefix + 'верхний лимит стоимости', '0', suffix))
    self.create_pattern('sum__lt_8', (prefix, 'максимум 0', suffix))
    self.create_pattern('sum__lt_9', (prefix, 'до 0', suffix))
    self.create_pattern('sum__lt_10', (prefix, 'но не превышающую 0', suffix))
    self.create_pattern('sum__lt_11', (prefix, 'совокупное пороговое значение 0', suffix))

    # greather than
    self.create_pattern('sum__gt_1', (prefix + 'составляет', 'более 0', suffix))
    self.create_pattern('sum__gt_2', (prefix + '', 'превышает 0', suffix))
    self.create_pattern('sum__gt_3', (prefix + '', 'свыше 0', suffix))
    self.create_pattern('sum__gt_4', (prefix + '', 'сделка имеет стоимость, равную или превышающую 0', suffix))



PricePF = PriceFactory(embedder)

"""### CharterPF"""

class CharterPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self.patterns_dict = {}

    #     self._build_paragraph_split_pattern()
    self._build_order_patterns()
    self._build_head_patterns()

    self.embedd()

  @deprecated
  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)
    
    head_prfx="СТАТЬЯ 0. "

    cp('d_competence', ('к', 'компетенции', 'чего-то относятся'))
    
    
    cp('d_head_all.5', (head_prfx, 'компетенции общего собрания акционеров\n', ''))
    cp('d_head_all.6', (head_prfx, 'собрание акционеров\n', ''))
    
    cp('d_head_all.0', ('', 'компетенции', ''))
    cp('d_head_all.2', ('', 'собрания', ''))
    cp('d_head_all.3', ('', 'участников', ''))
    cp('d_head_all.4', ('', 'акционеров', ''))
    
    
    cp('d_head_directors.3', (head_prfx, 'компетенция совета директоров\n', ''))
    cp('d_head_directors.3', ('', 'совет директоров общества\n', ''))
    cp('d_head_directors.0', ('', 'компетенции', ''))
    cp('d_head_directors.1', ('', 'совета', ''))
    cp('d_head_directors.2', ('', 'директоров', ''))
    
    
    cp('d_head_pravlenie.0', (head_prfx, 'компетенции правления', ''))
    cp('d_head_pravlenie.0', ('', 'компетенции', ''))
    cp('d_head_pravlenie.1', ('', 'правления', ''))
#     cp('d_head_pravlenie.2', ('', 'общества', ''))
    
    cp('d_head_gen.3', (head_prfx, 'компетенции генерального директора', ''))
    cp('d_head_gen.0', ('', 'компетенции', ''))
    cp('d_head_gen.1', ('', 'генерального', ''))
    cp('d_head_gen.2', ('', 'директора', ''))
                 
    
    
    cp('negation.1', ('', 'кроме', ''))
    cp('negation.2', ('', 'не', ''))
    cp('negation.3', ('за', 'исключением', ''))
    cp('negation.4', ('за', 'иные', 'вопросы'))
   
    
        
    cp('organs_1', ('\n', 'органы управления', '.\n органами управления общества являются'))


  @deprecated
  def _build_order_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    self.create_pattern('d_order_1', ('Порядок', 'одобрения сделок', 'в совершении которых имеется заинтересованность'))
    self.create_pattern('d_order_2', ('', 'принятие решений', 'о совершении сделок'))
    self.create_pattern('d_order_3', ('', 'одобрение заключения', 'изменения или расторжения какой-либо сделки Общества'))
    self.create_pattern('d_order_4', ('', 'Сделки', 'стоимость которой равна или превышает'))
    self.create_pattern('d_order_5', ('', 'Сделки', 'стоимость которой составляет менее'))

    

 


CharterPF = CharterPatternFactory(embedder)

"""## NER

### 2. NER Pattern factory
"""

class NerPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self.patterns_dict = {}

    self._build_ner_patterns()
    self.embedd()


  def _build_ner_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    for o_type in org_types.keys():
      cp(o_type, ('', org_types[o_type], '"'))


    cp('ner_org.1', ('Полное', 'фирменное наименование', 'общества на русском языке:'))
    

    cp('ner_org.6', ('', 'ОБЩИЕ ПОЛОЖЕНИЯ', ''))
    
    cp('ner_org.2', ('', 'фирменное', ''))
    cp('ner_org.3', ('', 'русском', ''))
    cp('ner_org.4', ('', 'языке', ''))
    cp('ner_org.5', ('', 'полное', ''))

    cp('nerneg_1', ('общество имеет', 'печать', ''))
    cp('nerneg_2', ('', 'сокращенное', ''))
    cp('nerneg_3', ('на', 'английском', 'языке'))


NerPF = NerPatternFactory(embedder)

"""### NER -- fallback method 
#### In memory of ஸ்ரீனிவாஸ ராமானுஜன் ஐயங்கார்
for the case headline is not found
"""

def _build_org_type_attention_vector(subdoc: CharterDocument):
  attention_vector_neg = make_soft_attention_vector(subdoc, 'nerneg_1', blur=80)
  attention_vector_neg = 1 + (1 - attention_vector_neg)  # normalize(attention_vector_neg * -1)
  return attention_vector_neg

"""### NER-2
based on detecting document structure and headlines
"""

# ------------------------------------------------------------------------------
def _detect_org_type_and_name(section, render=False):
  s_attention_vector_neg = _build_org_type_attention_vector(section)

  dict_org = {}
  best_type = None
  max = 0
  for org_type in org_types.keys():

    vector = section.distances_per_pattern_dict[org_type] * s_attention_vector_neg
    if render:
      print('_detect_org_type_and_name, org_type=', org_type, section.distances_per_pattern_dict[org_type][0:10])

    idx = np.argmax(vector)
    val = section.distances_per_pattern_dict[org_type][idx]
    if val > max:
      max = val
      best_type = org_type

    dict_org[org_type] = [idx, val]

  if render:
    print('_detect_org_type_and_name', dict_org)

  return dict_org, best_type



# ------------------------------------------------------------------------------
def detect_ners(section, render=False):
  assert section is not None
  
  section.embedd(NerPF)
  section.calculate_distances_per_pattern(NerPF)

  dict_org, best_type = _detect_org_type_and_name(section, render)

  if render:
    render_color_text(section.tokens_cc, section.distances_per_pattern_dict[best_type], _range=[0, 1])

  start = dict_org[best_type][0]
  start = start + len(NerPF.patterns_dict[best_type].embeddings)
  end = 1 + find_ner_end(section.tokens, start)

  orgname_sub_section = section.subdoc(start, end)
  org_name = untokenize(orgname_sub_section.tokens_cc)

  if render:
    render_color_text(orgname_sub_section.tokens_cc, orgname_sub_section.distances_per_pattern_dict[best_type],
                      _range=[0, 1])
    print('Org type:', org_types[best_type], dict_org[best_type])

  rez = {
    'type': best_type,
    'name': org_name,
    'type_name': org_types[best_type],
    'tokens': section.tokens_cc,
    'attention_vector': section.distances_per_pattern_dict[best_type]
  }

  return rez

"""## Margin (constraint) values detection

### Find constraint values

#### Rendering
"""

def sign_to_text(sign: int):
  if sign < 0: return " &lt; "
  if sign > 0: return " &gt; "
  return ' = '

def value_to_html(vc: ValueConstraint):
  color = '#333333'
  if vc.sign > 0:
    color = '#993300'
  elif vc.sign < 0:
    color = '#009933'

  return f'<b style="color:{color}">{sign_to_text(vc.sign)} {vc.currency} {vc.value:20,.2f}</b> '


def _render_sentence(sentence):
  html = ""
  constraints:List[ValueConstraint] = sentence['constraints']
  for c in constraints:
    html += value_to_html(c)

  if len(constraints) > 0:
    html += '<div style="border-bottom:1px solid #ccc; margin-top:1em"></div>'
    section = sentence['subdoc']
    html += to_color_text(section.tokens, section.distances_per_pattern_dict['deal_value_attention_vector'])
  return html



def render_constraint_values(rz):
  html = ''
  for head_type in rz.keys():

    r_by_head_type = rz[head_type]

    html += '<hr style="margin-top: 45px">'
    html += '<i style="padding:0; margin:0">решения о пороговых суммах, которые принимает</i><h2 style="color:{}; padding:0;margin:0">{}</h2>'.format(
      head_types_colors[head_type],
      head_types_dict[head_type])

    sentences = r_by_head_type['sentences']
    html += '<h4>{}</h4>'.format(r_by_head_type['caption'])
    html += '<div style="padding-left:80px">'

    if True:
      if len(sentences) > 0:
        for sentence in sentences:
          html += _render_sentence(sentence)

      else:
        html += '<h4 style="color:crimson">Пороговые суммы не найдены или не заданы</h4>'

    html += '</div>'

  return html

"""### extract_constraint_values_from_region"""

from typing import List

import nltk

from embedding_tools import embedd_tokenized_sentences_list
from legal_docs import LegalDocument, AbstractPatternFactory, extract_all_contraints_from_sentence
from charter_patterns import make_constraints_attention_vectors
from text_tools import untokenize
from transaction_values import extract_sum, ValueConstraint


def split_by_token(tokens: List[str], token):
  res = []
  sentence = []
  for i in tokens:
    if i == token:
      res.append(sentence)
      sentence = []
    else:
      sentence.append(i)
  
  res.append(sentence)
  return res


def _extract_constraint_values_from_region(sentenses_i, _embedd_factory, render=False):
  if sentenses_i is None or len(sentenses_i)==0:
    return []
  
  ssubdocs = embedd_generic_tokenized_sentences(sentenses_i, _embedd_factory)

  for ssubdoc in ssubdocs:

    vectors = make_constraints_attention_vectors(ssubdoc)
    ssubdoc.distances_per_pattern_dict = {**ssubdoc.distances_per_pattern_dict, **vectors}

    if render:
      render_color_text(
        ssubdoc.tokens,
        ssubdoc.distances_per_pattern_dict['deal_value_attention_vector'], _range=(0, 1))

  sentences = []
  for sentence_subdoc in ssubdocs:
    constraints: List[ValueConstraint] = extract_all_contraints_from_sentence(sentence_subdoc,
                                                                              sentence_subdoc.distances_per_pattern_dict[
                                                                                'deal_value_attention_vector'])

    sentence = {
      'quote': untokenize(sentence_subdoc.tokens_cc),
      'subdoc': sentence_subdoc,
      'constraints': constraints
    }

    sentences.append(sentence)
  return sentences


def embedd_generic_tokenized_sentences(strings: List[str], factory: AbstractPatternFactory) -> \
        List[LegalDocument]:
  embedded_docs = []
  if strings is None or len(strings)==0:
    return []

  tokenized_sentences_list = []
  for i in range(len(strings)):
    s = strings[i]

    words = nltk.word_tokenize(s)

    subdoc = LegalDocument()

    subdoc.tokens = words
    subdoc.tokens_cc = words

    tokenized_sentences_list.append(subdoc.tokens)
    embedded_docs.append(subdoc)

  sentences_emb, wrds, lens = embedd_tokenized_sentences_list(factory.embedder, tokenized_sentences_list)

  for i in range(len(embedded_docs)):
    l = lens[i]
    tokens = wrds[i][:l]

    line_emb = sentences_emb[i][:l]

    embedded_docs[i].tokens = tokens
    embedded_docs[i].tokens_cc = tokens
    embedded_docs[i].embeddings = line_emb
    embedded_docs[i].calculate_distances_per_pattern(factory)

  return embedded_docs


##---------------------------------------
def extract_constraint_values_from_section(section, verbose=False):
  _embedd_factory = PricePF

  if verbose:
    print('extract_constraint_values_from_sections', section['headline.type'])

  body = section['body.subdoc']

  if verbose:
    print('extract_constraint_values_from_sections', 'embedding....')

  sentenses_i = []
  senetences = split_by_token(body.tokens, '\n')
  for s in senetences:
    line = untokenize(s) + '\n'
    sum = extract_sum(line)
    if sum is not None:
      sentenses_i.append(line)
    if verbose:
      print('-', sum, line)

  hl_subdoc = section['headline.subdoc']

  r_by_head_type = {
    'section': head_types_dict[section['headline.type']],
    'caption': untokenize(hl_subdoc.tokens_cc),
    'sentences': _extract_constraint_values_from_region(sentenses_i, _embedd_factory, render=verbose)
  }

  return r_by_head_type


##---------------------------------------


def extract_constraint_values_from_sections(sections, verbose=False):
  rez = {}

  for head_type in sections:
    section = sections[head_type]
    rez[head_type] = extract_constraint_values_from_section(section, verbose)

  return rez

"""## ~ALL together~"""

# MAIN METHOD
import gc

from legal_docs import CharterDocument, embedd_headlines

gc.collect()

import numpy as np


# ---------------------------------------
def find_contraints(sections, verbose=False):
 

  # 5. extract constraint values
  sections_filtered = {}
  prefix = 'head.'
  for k in sections:
    if k[:len(prefix)] == prefix:
      sections_filtered[k] = sections[k]
      

  rz = extract_constraint_values_from_sections(sections_filtered)
  return rz


# ---------------------------------------
def process_charter(txt, verbose=False ):
  # parse
  _charter_doc = CharterDocument(txt)
  _charter_doc.right_padding = 0
  _charter_doc.parse()

  # 1. find top level structure
  #   headline_indexes = _charter_doc.structure.get_lines_by_level(0)

  headline_indexes = _charter_doc.structure.headline_indexes

  # 2. embedd headlines
  embedded_headlines = embedd_headlines(headline_indexes, _charter_doc, HPF)

  # 3. apply semantics to headlines,
  best_indexes = match_headline_types(HPF.headlines, headline_indexes, embedded_headlines, 'headline.', 1.4)
  
  # 4. find sections
  sections = find_sections_by_headlines(best_indexes,
                                        _charter_doc,
                                        headline_indexes,
                                        render=verbose)

  #   org_subdoc = _doc_section_under_headline(_charter_doc, hl_struct, _headline_indexes, embedd_factory=NerPF, render=render)
  #   _org = detect_ners(section=org_subdoc)

  if 'name' in sections:
    org = detect_ners(section=sections['name']['body.subdoc'], render=verbose)
  else:
    org = {
      'type': 'org_unknown',
      'name': "не определено",
      'type_name': "не определено",
      'tokens': [],
      'attention_vector': []
    }

  rz  = find_contraints(sections, verbose)

  #   html = render_constraint_values(rz)
  #   display(HTML(html))

  return org, rz


# -------------------------------------------------------------------------------------

# RENDER
def render_charter_parsing_results(org, rz):
  txt_html = to_color_text(org['tokens'], org['attention_vector'], _range=[0, 1])

  html = '<div style="background:#eeeeff; padding:0.5em"> recognized NE(s): <br><br> org type:<h3 style="margin:0">  {} </h3>org full name:<h2 style="margin:0">  {} </h2> <br>quote: <div style="font-size:90%; background:white">{}</div> </div>'.format(
    org['type_name'], org['name'], txt_html)
  # html+=txt_html
  html += render_constraint_values(rz)

  display(HTML(html))


# -------------------------------------------------------------------------------------
if dev_mode:
  # TESTING
  # -------------------------------------------------------------------------------------

  org, rz = process_charter(TEST_CHARTER_TEXT, verbose=True)
  render_charter_parsing_results(org, rz)
  gc.collect()

"""# DEMO

## Upload
"""

tf.logging.set_verbosity('FATAL')

print_text = False  #@param {type: "boolean"}
analyze_immediately_after_upload = True  #@param {type: "boolean"}

docs=[]
if upload_enabled:
  #--------------------------
  docs = interactive_upload()
  #--------------------------


  if print_text:
    print(docs[0])
  
  
  if analyze_immediately_after_upload and upload_enabled:
    txt = docs[0]
    tf.logging.set_verbosity('FATAL')
    org, rz = process_charter(txt, verbose=False)
    html = render_charter_parsing_results(org, rz)



"""## Analyze"""

if upload_enabled:
  txt = docs[0]
  
  org, rz = process_charter(txt, verbose=True)
  render_charter_parsing_results(org, rz)

render_charter_parsing_results(org, rz)