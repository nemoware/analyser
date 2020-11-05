from overrides import overrides

from analyser.contract_agents import ContractAgent, normalize_contract_agent
from analyser.doc_dates import find_date
from analyser.documents import TextMap
from analyser.legal_docs import LegalDocument, ContractValue, ParserWarnings
from analyser.log import logger
from analyser.ml_tools import *
from analyser.parsing import ParsingContext, AuditContext, find_value_sign_currency_attention
from analyser.patterns import AV_SOFT, AV_PREFIX
from analyser.text_tools import find_top_spans
from tf_support.tf_subject_model import load_subject_detection_trained_model, decode_subj_prediction, \
  nn_predict


class ContractDocument(LegalDocument):

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.subjects: SemanticTag or None = None
    self.contract_values: List[ContractValue] = []

    self.agents_tags: [SemanticTag] = []

  def get_tags(self) -> [SemanticTag]:
    tags = []
    if self.date is not None:
      tags.append(self.date)

    if self.number is not None:
      tags.append(self.number)

    if self.agents_tags:
      tags += self.agents_tags

    if self.subjects:
      tags.append(self.subjects)

    if self.contract_values:
      for contract_value in self.contract_values:
        tags += contract_value.as_list()

    # TODO: filter tags if _t.isNotEmpty():
    return tags


ContractDocument3 = ContractDocument


class ContractParser(ParsingContext):

  def __init__(self, embedder=None, sentence_embedder=None):
    ParsingContext.__init__(self, embedder, sentence_embedder)
    self.subject_prediction_model = load_subject_detection_trained_model()

  def init_embedders(self, embedder, elmo_embedder_default):
    # TODO: remove
    warnings.warn('init_embedders will be removed in future versions, embbeders will be lazyly inited on demand',
                  DeprecationWarning)

  def find_org_date_number(self, doc: ContractDocument, ctx: AuditContext) -> ContractDocument:

    contract = doc[0:300]  # warning, trimming doc for analysis phase 1
    if contract.embeddings is None:
      logger.debug('embedding 300-trimmed contract')
      contract.embedd_tokens(self.get_embedder())

    # predicting with NN
    logger.debug('predicting semantic_map in 300-trimmed contract with NN')
    semantic_map, _ = nn_predict(self.subject_prediction_model, contract)

    doc.agents_tags = nn_find_org_names(contract.tokens_map, semantic_map,
                                        audit_subsidiary_name=ctx.audit_subsidiary_name)

    # TODO: maybe move contract.tokens_map into text map
    doc.number = nn_get_contract_number(contract.tokens_map, semantic_map)
    doc.date = nn_get_contract_date(contract.tokens_map, semantic_map)

    return doc

  def validate(self, document: ContractDocument, ctx: AuditContext):
    document.clear_warnings()

    if not document.date:
      document.warn(ParserWarnings.date_not_found)

    if not document.number:
      document.warn(ParserWarnings.number_not_found)

    if not document.contract_values:
      document.warn(ParserWarnings.contract_value_not_found)

    if not document.subjects:
      document.warn(ParserWarnings.contract_subject_not_found)

    self.log_warnings()

  @overrides
  def find_attributes(self, contract: ContractDocument, ctx: AuditContext) -> ContractDocument:
    """
    this analyser should care about embedding, because it decides wheater it needs (NN) embeddings or not  
    """

    self._reset_context()

    _contract_cut = contract
    if len(contract) > HyperParameters.max_doc_size_tokens:
      contract.warn_trimmed(HyperParameters.max_doc_size_tokens)
      _contract_cut = contract[0:HyperParameters.max_doc_size_tokens]  # warning, trimming doc for analysis phase 1

    # ------ lazy embedding
    if _contract_cut.embeddings is None:
      _contract_cut.embedd_tokens(self.get_embedder())

    # self.find_org_date_number(_contract_cut, ctx)

    semantic_map, subj_1hot = nn_predict(self.subject_prediction_model, _contract_cut)

    # repeat phase 1
    # ---

    if not contract.number:
      contract.number = nn_get_contract_number(_contract_cut.tokens_map, semantic_map)

    if not contract.date:
      contract.date = nn_get_contract_date(_contract_cut.tokens_map, semantic_map)

    if len(contract.agents_tags) < 2:
      contract.agents_tags = nn_find_org_names(_contract_cut.tokens_map, semantic_map,
                                               audit_subsidiary_name=ctx.audit_subsidiary_name)
    # -------------------------------subject
    contract.subjects = nn_get_subject(_contract_cut.tokens_map, semantic_map, subj_1hot)

    # -------------------------------values
    contract.contract_values = nn_find_contract_value(_contract_cut, semantic_map)

    self._logstep("finding contract values")
    # --------------------------------------

    self.validate(contract, ctx)

    return contract


# --------------- END of CLASS


ContractAnlysingContext = ContractParser  ##just alias, for ipnb compatibility. TODO: remove


def max_confident(vals: List[ContractValue]) -> ContractValue:
  if len(vals) == 0:
    return None
  return max(vals, key=lambda a: a.integral_sorting_confidence())


def max_value(vals: List[ContractValue]) -> ContractValue:
  if len(vals) == 0:
    return None
  return max(vals, key=lambda a: a.value.value)


def _sub_attention_names(subj: Enum):
  a = f'x_{subj}'
  b = AV_PREFIX + f'x_{subj}'
  c = AV_SOFT + a
  return a, b, c


def nn_find_org_names(textmap: TextMap, semantic_map: DataFrame,
                      audit_subsidiary_name=None) -> [SemanticTag]:
  cas = []
  for o in [1, 2]:
    ca = ContractAgent()
    for n in ['name', 'alias', 'type']:
      tagname = f'org-{o}-{n}'
      tag = nn_get_tag_value(tagname, textmap, semantic_map)
      setattr(ca, n, tag)
    normalize_contract_agent(ca)
    cas.append(ca)

  def name_val_safe(a):
    if a.name is not None:
      return a.name.value
    return ''

  if audit_subsidiary_name:
    # known subsidiary goes first
    cas = sorted(cas, key=lambda a: name_val_safe(a) != audit_subsidiary_name)
  else:
    cas = sorted(cas, key=lambda a: name_val_safe(a))

  return _swap_org_tags(cas)


def _swap_org_tags(all_: [ContractAgent]) -> [SemanticTag]:
  tags = []
  for n, agent in enumerate(all_):
    for tag in agent.as_list():
      tagname = f"org-{n + 1}-{tag.kind.split('-')[-1]}"
      tag.kind = tagname
      tags.append(tag)

  return tags


def nn_find_contract_value(contract: ContractDocument, semantic_map: DataFrame) -> [ContractValue]:
  _keys = ['sign_value_currency/value', 'sign_value_currency/currency', 'sign_value_currency/sign']
  attention_vector = semantic_map[_keys].values.sum(axis=-1)

  values_list: [ContractValue] = find_value_sign_currency_attention(contract, attention_vector)
  if len(values_list) == 0:
    return []
  # ------
  # reduce number of found values
  # take only max value and most confident ones (we hope, it is the same finding)

  max_confident_cv: ContractValue = max_confident(values_list)
  if max_confident_cv.value.confidence < 0.1:
    return []

  return [max_confident_cv]

  # max_valued_cv: ContractValue = max_value(values_list)
  # if max_confident_cv == max_valued_cv:
  #   return [max_confident_cv]
  # else:
  #   # TODO: Insurance docs have big value, its not what we're looking for. Biggest is not the best see https://github.com/nemoware/analyser/issues/55
  #   # TODO: cannot compare diff. currencies
  #   max_valued_cv *= 0.5
  #   return [max_valued_cv]


def nn_get_subject(textmap: TextMap, semantic_map: DataFrame, subj_1hot) -> SemanticTag:
  predicted_subj_name, confidence, _ = decode_subj_prediction(subj_1hot)

  tag = SemanticTag('subject', predicted_subj_name.name, span=None)
  tag.confidence = confidence

  tag_ = nn_get_tag_value('subject', textmap, semantic_map)
  if tag_ is not None:
    tag.span = tag_.span

  return tag


def nn_get_contract_number(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tag = nn_get_tag_value('number', textmap, semantic_map)
  if tag is not None:
    tag.value = tag.value.strip().lstrip('№').lstrip().lstrip(':').lstrip('N ').lstrip().rstrip('.')
    nn_fix_span(tag)
  return tag


def nn_get_contract_date(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tag = nn_get_tag_value('date', textmap, semantic_map)
  if tag is not None:
    _, dt = find_date(tag.value)
    tag.value = dt
    if dt is not None:
      return tag


def nn_get_tag_value(tagname: str, textmap: TextMap, semantic_map: DataFrame, threshold=0.3) -> SemanticTag or None:
  att = semantic_map[tagname].values
  slices = find_top_spans(att, threshold=threshold, limit=1)  # TODO: estimate per-tag thresholds

  if len(slices) > 0:
    span = slices[0].start, slices[0].stop
    value = textmap.text_range(span)
    tag = SemanticTag(tagname, value, span)
    tag.confidence = float(att[slices[0]].mean())
    return tag
  return None


def nn_fix_span(tag: SemanticTag):
  return tag
