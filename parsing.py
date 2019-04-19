import time
from functools import wraps

from renderer import AbstractRenderer

PROF_DATA = {}


class ParsingConfig:
  def __init__(self):
    self.headline_attention_threshold=1.0

class ParsingContext:
  def __init__(self, embedder, renderer: AbstractRenderer):
    assert embedder is not None
    assert renderer is not None
    self.renderer = renderer
    self.embedder = embedder
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
