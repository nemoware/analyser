import time
from functools import wraps

from analyser.legal_docs import LegalDocument
from analyser.structures import ContractSubject

PROF_DATA = {}


class ParsingSimpleContext:
  def __init__(self):

    # ---
    self.verbosity_level = 2
    self.__step = 0

    self.warnings = []


  def _reset_context(self):
    self.warnings = []
    self.__step = 0

  def _logstep(self, name: str) -> None:
    s = self.__step
    print(f'❤️ ACCOMPLISHED: \t {s}.\t {name}')
    self.__step += 1

  def warning(self, text):
    t_ = '⚠️ WARNING: - ' + text
    self.warnings.append(t_)
    print(t_)

  def get_warings(self):
    return '\n'.join(self.warnings)

  def log_warnings(self):
    if len(self.warnings) > 0:
      print("Recent parsing warnings:")

      for w in self.warnings:
        print('\t\t', w)


class AuditContext:

  def __init__(self):
    self.audit_subsidiary_name:str=None


class ParsingContext(ParsingSimpleContext):
  def __init__(self, embedder=None):
    ParsingSimpleContext.__init__(self)
    self.embedder = embedder

  def init_embedders(self, embedder, elmo_embedder_default):
    raise NotImplementedError()

  def find_org_date_number(self, document: LegalDocument, ctx:AuditContext) -> LegalDocument:
    """
    phase 1, before embedding TF, GPU, and things
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    raise NotImplementedError()

  def find_attributes(self, document: LegalDocument):
    raise NotImplementedError()


def profile(fn):
  @wraps(fn)
  @wraps(fn)
  def with_profiling(*args, **kwargs):
    start_time = time.time()

    ret = fn(*args, **kwargs)

    elapsed_time = time.time() - start_time

    if fn.__name__ not in PROF_DATA:
      PROF_DATA[fn.__name__] = [0, []]
    PROF_DATA[fn.__name__][0] += 1
    PROF_DATA[fn.__name__][1].append(elapsed_time)

    return ret

  return with_profiling


def print_prof_data():
  for fname, data in PROF_DATA.items():
    max_time = max(data[1])
    avg_time = sum(data[1]) / len(data[1])
    print("Function {} called {} times. ".format(fname, data[0]))
    print('Execution time max: {:.4f}, average: {:.4f}'.format(max_time, avg_time))


def clear_prof_data():
  global PROF_DATA
  PROF_DATA = {}


head_types_dict = {'head.directors': 'Совет директоров',
                   'head.all': 'Общее собрание участников/акционеров',
                   'head.gen': 'Генеральный директор',
                   #                      'shareholders':'Общее собрание акционеров',
                   'head.pravlenie': 'Правление общества',
                   'head.unknown': '*Неизвестный орган управления*'}
head_types = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie']

known_subjects = [
  ContractSubject.Charity,
  ContractSubject.RealEstate,
  ContractSubject.Lawsuit]
