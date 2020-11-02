from analyser.attributes import list_tags, copy_leaf_tag


class AMatcher():
  def matches(self, path: [str]) -> bool:
    raise NotImplementedError()

  def convert(self, path: [str], attr, dest):
    raise NotImplementedError()


class Converter():
  matchers: [AMatcher] = []

  def register(self, m: AMatcher):
    self.matchers.append(m)

  def convert(self, attrs):
    for path, v in list_tags(attrs):
      for m in self.matchers:
        if m.matches(path):
          m.convert(path, v, )


class DateMatcher(AMatcher):

  def matches(self, path: [str]) -> bool:
    return len(path) == 1 and (path[0] in ['date', 'number'])

  def convert(self, path: [str], attr, dest):
    attr_name: str = path[0]
    copy_leaf_tag(attr_name, attr, dest)
