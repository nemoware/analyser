import glob
from typing import List, Generator


class AbstractDocProvider:

  def getdocuments(self, query) -> Generator:
    raise NotImplementedError()

  def list_filenames(self, filename_prefix) -> List[str]:
    raise NotImplementedError()


class DirDocProvider(AbstractDocProvider):
  def list_filenames(self, filename_prefix) -> List[str]:
    filenames = []
    filenames += [file for file in glob.glob(filename_prefix + "**/*.docx", recursive=True)]
    filenames += [file for file in glob.glob(filename_prefix + "**/*.doc", recursive=True)]
    return sorted(filenames)
