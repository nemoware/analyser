from typing import List

from legal_docs import LegalDocument, deprecated
from parsing import ParsingContext
from patterns import AbstractPatternFactory


class SectionsFinder:

  def __init__(self, ctx: ParsingContext):
    self.ctx = ctx
    pass

  def find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.') -> dict:
    raise NotImplementedError()


class DefaultSectionsFinder(SectionsFinder):
  def __init__(self, ctx: ParsingContext):
    SectionsFinder.__init__(self, ctx)

  @deprecated
  def find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.') -> dict:
    embedded_headlines = doc.embedd_headlines(factory)

    doc.sections = doc.find_sections_by_headlines_2(
      self.ctx, headlines, embedded_headlines, headline_patterns_prefix, self.ctx.config.headline_attention_threshold)

    self.ctx._logstep("embedding headlines into semantic space")

    return doc.sections
