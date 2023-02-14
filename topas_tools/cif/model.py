from future import standard_library
standard_library.install_aliases()
from past.builtins import basestring
from libtbx.containers import OrderedDict, OrderedSet
from libtbx.utils import Sorry
import sys
import string
import copy
from io import StringIO

from cctbx.array_family import flex


from collections.abc import MutableMapping

class DictMixin(MutableMapping):
    def __len__(self):
        return len(self.mylist)

    def __iter__(self):
        for i in self.mylist:
            yield i



class cif(DictMixin):
  def __init__(self, blocks=None):
    if blocks is not None:
      self.blocks = OrderedDict(blocks)
    else:
      self.blocks = OrderedDict()
    self.keys_lower = {key.lower(): key for key in list(self.blocks.keys())}

  def __setitem__(self, key, value):
    assert isinstance(value, block)
    if not re.match(tag_re, '_'+key):
      raise Sorry("%s is not a valid data block name" %key)
    self.blocks[key] = value
    self.keys_lower[key.lower()] = key

  def get(self, key, default=None):
    key_lower = self.keys_lower.get(key.lower())
    if (key_lower is None):
      return default
    return self.blocks.get(key_lower, default)

  def __getitem__(self, key):
    result = self.get(key)
    if (result is None):
      raise KeyError('Unknown CIF data block name: "%s"' % key)
    return result

  def __delitem__(self, key):
    del self.blocks[self.keys_lower[key.lower()]]
    del self.keys_lower[key.lower()]

  def keys(self):
    return list(self.blocks.keys())

  def __repr__(self):
    return repr(OrderedDict(iter(self.items())))

  def __copy__(self):
    return cif(self.blocks.copy())

  copy = __copy__

  def __deepcopy__(self, memo):
    return cif(copy.deepcopy(self.blocks, memo))

  def deepcopy(self):
    return copy.deepcopy(self)

  def show(self, out=None, indent="  ", indent_row=None,
           data_name_field_width=34,
           loop_format_strings=None):
    if out is None:
      out = sys.stdout
    for name, block in list(self.items()):
      print("data_%s" %name, file=out)
      block.show(
        out=out, indent=indent, indent_row=indent_row,
        data_name_field_width=data_name_field_width,
        loop_format_strings=loop_format_strings)

  def __str__(self):
    s = StringIO()
    self.show(out=s)
    return s.getvalue()

  def validate(self, dictionary, show_warnings=True, error_handler=None, out=None):
    if out is None: out = sys.stdout
    from . import validation
    errors = {}
    if error_handler is None:
      error_handler = validation.ErrorHandler()
    for key, block in self.blocks.items():
      error_handler = error_handler.__class__()
      dictionary.set_error_handler(error_handler)
      block.validate(dictionary)
      errors.setdefault(key, error_handler)
      if error_handler.error_count or error_handler.warning_count:
        error_handler.show(show_warnings=show_warnings, out=out)
    return error_handler

  def sort(self, recursive=False, key=None, reverse=False):
    self.blocks = OrderedDict(sorted(list(self.blocks.items()), key=key, reverse=reverse))
    if recursive:
      for b in list(self.blocks.values()):
        b.sort(recursive=recursive, reverse=reverse)

class block_base(DictMixin):
  def __init__(self):
    self._items = {}
    self.loops = {}
    self._set = OrderedSet()
    self.keys_lower = {}

  def __setitem__(self, key, value):
    if not re.match(tag_re, key):
      raise Sorry("%s is not a valid data name" %key)
    if isinstance(value, loop):
      self.loops[key] = value
      for k in list(value.keys()):
        self.keys_lower[k.lower()] = k
    elif isinstance(value, basestring):
      v = str(value)
      if not (re.match(any_print_char_re, v) or
              re.match(quoted_string_re, v) or
              re.match(semicolon_string_re, v)):
        raise Sorry("Invalid data item for %s" %key)
      self._items[key] = v
      self.keys_lower[key.lower()] = key
    else:
      try:
        float(value)
        self[key] = str(value)
      except TypeError:
        if key in self._items:
          del self._items[key]
        for loop_ in list(self.loops.values()):
          if key in loop_:
            loop_[key] = value
        if key not in self:
          self.add_loop(loop(header=(key,), data=(value,)))
    if key in self._items or isinstance(value, loop):
      self._set.add(key)

  def __getitem__(self, key):
    key = self.keys_lower.get(key.lower(), key)
    if key in self._items:
      return self._items[key]
    else:
      # give precedence to returning the actual data items in the event of a
      # single looped item when the loop name and data name coincide
      for loop in list(self.loops.values()):
        if key in loop:
          return loop[key]
      if key in self.loops:
        return self.loops[key]
    raise KeyError

  def __delitem__(self, key):
    key = self.keys_lower.get(key.lower(), key)
    if key in self._items:
      del self._items[key]
      self._set.discard(key)
    elif key in list(self.keys()):
      # must be a looped item
      for k, loop in self.loops.items():
        if key in loop:
          if len(loop) == 1:
            # remove the now empty loop
            del self[k]
          else:
            del loop[key]
          return
      raise KeyError
    elif key in self.loops:
      del self.loops[key]
      self._set.discard(key)
    else:
      raise KeyError

  def get_looped_item(self,
                      key,
                      key_error=KeyError,
                      value_error=ValueError,
                      default=None):
    if key not in self:
      if key_error is None:
        return default
      else:
        raise key_error(key)
    value = self[key]
    if isinstance(value, flex.std_string):
      return value
    elif value_error is not None:
      raise value_error("%s is not a looped item" %key)
    else:
      return default

  def loop_keys(self):
    done = []
    for key in self:
      key = key.split(".")[0]
      if key in done: continue
      done.append(key)
    return done

  def iterloops(self):
    for key in self.loop_keys():
      yield self.get(key)

  def get_single_item(self,
                      key,
                      key_error=KeyError,
                      value_error=ValueError,
                      default=None):
    if key not in self:
      if key_error is None:
        return default
      else:
        raise key_error(key)
    value = self[key]
    if not isinstance(value, flex.std_string):
      return value
    elif value_error is not None:
      raise value_error("%s appears as a looped item" %key)
    else:
      return default

  def keys(self):
    keys = []
    for key in self._set:
      if key in self._items:
        keys.append(key)
      elif key in self.loops:
        keys.extend(list(self.loops[key].keys()))
    return keys

  def __repr__(self):
    return repr(OrderedDict(iter(self.items())))

  def update(self, other=None, **kwargs):
    if other is None:
      return
    if isinstance(other, OrderedDict) or isinstance(other, dict):
      for key, value in other.items():
        self[key] = value
    else:
      self._items.update(other._items)
      self.loops.update(other.loops)
      self._set |= other._set
      self.keys_lower.update(other.keys_lower)

  def add_data_item(self, tag, value):
    self[tag] = value

  def add_loop(self, loop):
    try:
      self.setdefault(loop.name(), loop)
    except Sorry:
      # create a unique loop name
      self.setdefault('_'+str(hash(tuple(loop.keys()))), loop)

  def get_loop(self, loop_name, default=None):
    loop_ = self.loops.get(loop_name.lower())
    if loop_ is None:
      return default
    return loop_

  def get_loop_with_defaults(self, loop_name, default_dict):
    loop_ = self.get_loop(loop_name)
    if loop_ is None:
      loop_ = loop(header=list(default_dict.keys()))
    n_rows = loop_.n_rows()
    for key, value in default_dict.items():
      if key not in loop_:
        loop_.add_column(key, flex.std_string(n_rows, value))
    return loop_

  def __copy__(self):
    new = self.__class__()
    new._items = self._items.copy()
    new.loops = self.loops.copy()
    new._set = copy.copy(self._set)
    new.keys_lower = self.keys_lower.copy()
    return new

  copy = __copy__

  def __deepcopy__(self, memo):
    new = self.__class__()
    new._items = copy.deepcopy(self._items, memo)
    new.loops = copy.deepcopy(self.loops, memo)
    new._set = copy.deepcopy(self._set, memo)
    new.keys_lower = copy.deepcopy(self.keys_lower, memo)
    return new

  def deepcopy(self):
    return copy.deepcopy(self)

  def __str__(self):
    s = StringIO()
    self.show(out=s)
    return s.getvalue()

  def validate(self, dictionary):
    for key, value in self._items.items():
      dictionary.validate_single_item(key, value, self)
    for loop in list(self.loops.values()):
      dictionary.validate_loop(loop, self)
    if isinstance(self, block):
      for value in self.saves.values():
        value.validate(dictionary)

  def sort(self, recursive=False, key=None, reverse=False):
    self._set = OrderedSet(
      sorted(list(self._items.keys()), key=key, reverse=reverse) \
      + sorted(list(self.loops.keys()), key=key, reverse=reverse))
    if recursive:
      for l in list(self.loops.values()):
        l.sort(key=key, reverse=reverse)

  """Items that either appear in both self and other and the value has changed
     or appear in self but not other."""
  def difference(self, other):
    new = self.__class__()
    for items in (self._items, self.loops):
      for key, value in items.items():
        if key in other:
          other_value = other[key]
          if other_value == value: continue
          else:
            new[key] = other_value
        else:
          new[key] = value
    return new

class save(block_base):

  def show(self, out=None, indent="  ", data_name_field_width=34):
    if out is None:
      out = sys.stdout
      assert indent.strip() == ""
    format_str = "%%-%is" %(data_name_field_width-1)
    for k in self._set:
      v = self._items.get(k)
      if v is not None:
        print(indent + format_str %k, format_value(v), file=out)
      else:
        print(indent, end=' ', file=out)
        self.loops[k].show(out=out, indent=(indent+indent))
        print(file=out)


class block(block_base):

  def __init__(self):
    block_base.__init__(self)
    self.saves = {}

  def __setitem__(self, key, value):
    if isinstance(value, save):
      self.saves[key] = value
      self.keys_lower[key.lower()] = key
      self._set.add(key)
    else:
      block_base.__setitem__(self, key, value)

  def __getitem__(self, key):
    key = self.keys_lower.get(key.lower(), key)
    if key in self.saves:
      return self.saves[key]
    else:
      return block_base.__getitem__(self, key)

  def __delitem__(self, key):
    key = self.keys_lower.get(key.lower(), key)
    if key in self.saves:
      del self.saves[key]
      self._set.discard(key)
    else:
      block_base.__delitem__(self, key)

  def keys(self):
    keys = block_base.keys(self)
    keys.extend(list(self.saves.keys()))
    return keys

  def update(self, other=None, **kwargs):
    if other is None:
      return
    block_base.update(self, other, **kwargs)
    self.saves.update(other.saves)

  def show(self, out=None, indent="  ", indent_row=None,
           data_name_field_width=34,
           loop_format_strings=None):
    assert indent.strip() == ""
    if out is None:
      out = sys.stdout
    format_str = "%%-%is" %(data_name_field_width-1)
    for k in self._set:
      v = self._items.get(k)
      if v is not None:
        print(format_str %k, format_value(v), file=out)
      elif k in self.saves:
        print(file=out)
        print("save_%s" %k, file=out)
        self.saves[k].show(out=out, indent=indent,
                           data_name_field_width=data_name_field_width)
        print(indent + "save_", file=out)
        print(file=out)
      else:
        if loop_format_strings is not None and k in loop_format_strings:
          self.loops[k].show(
            out=out, indent=indent, indent_row=indent_row,
            fmt_str=loop_format_strings[k])
        else:
          self.loops[k].show(out=out, indent=indent, indent_row=indent_row)
        print(file=out)

  def sort(self, recursive=False, key=None, reverse=False):
    block_base.sort(self, recursive=recursive, key=key, reverse=reverse)
    if recursive:
      for s in list(self.saves.values()):
        s.sort(recursive=recursive, key=key, reverse=reverse)

  def __deepcopy__(self, memo):
    new = block_base.__deepcopy__(self, memo)
    new.saves = copy.deepcopy(self.saves, memo)
    return new

  def __copy__(self):
    new = block_base.copy(self)
    new.saves = self.saves.copy()
    return new

class loop(DictMixin):
  def __init__(self, header=None, data=None):
    self._columns = OrderedDict()
    self.keys_lower = {}
    if header is not None:
      for key in header:
        self.setdefault(key, flex.std_string())
      if data is not None:
        # the number of data items must be an exact multiple of the number of headers
        assert len(data) % len(header) == 0, "Wrong number of data items for loop"
        n_rows = len(data)//len(header)
        n_columns = len(header)
        for i in range(n_rows):
          self.add_row([data[i*n_columns+j] for j in range(n_columns)])
    elif header is None and data is not None:
      assert isinstance(data, dict) or isinstance(data, OrderedDict)
      self.add_columns(data)
      self.keys_lower = {
        key.lower(): key for key in list(self._columns.keys())}

  def __setitem__(self, key, value):
    if not re.match(tag_re, key):
      raise Sorry("%s is not a valid data name" %key)
    if len(self) > 0:
      assert len(value) == self.size()
    if not isinstance(value, flex.std_string):
      for flex_numeric_type in (flex.int, flex.double):
        if isinstance(value, flex_numeric_type):
          value = value.as_string()
        else:
          try:
            value = flex_numeric_type(value).as_string()
          except TypeError:
            continue
          else:
            break
      if not isinstance(value, flex.std_string):
        value = flex.std_string(value)
    # value must be a mutable type
    assert hasattr(value, '__setitem__')
    self._columns[key] = value
    self.keys_lower[key.lower()] = key

  def __getitem__(self, key):
    return self._columns[self.keys_lower[key.lower()]]

  def __delitem__(self, key):
    del self._columns[self.keys_lower[key.lower()]]
    del self.keys_lower[key.lower()]

  def keys(self):
    return list(self._columns.keys())

  def __repr__(self):
    return repr(OrderedDict(iter(self.items())))

  def name(self):
    return common_substring(list(self.keys())).rstrip('_').rstrip('.')

  def size(self):
    size = 0
    for column in list(self.values()):
      size = max(size, len(column))
    return size

  def n_rows(self):
    size = 0
    for column in list(self.values()):
      size = max(size, len(column))
    return size

  def n_columns(self):
    return len(list(self.keys()))

  def add_row(self, row, default_value="?"):
    if isinstance(row, dict):
      for key in self:
        if key in row:
          self[key].append(str(row[key]))
        else:
          self[key].append(default_value)
    else:
      assert len(row) == len(self)
      for i, key in enumerate(self):
        self[key].append(str(row[i]))

  def add_column(self, key, values):
    if self.size() != 0:
      assert len(values) == self.size()
    self[key] = values
    self.keys_lower[key.lower()] = key

  def add_columns(self, columns):
    assert isinstance(columns, dict) or isinstance(columns, OrderedDict)
    for key, value in columns.items():
      self.add_column(key, value)

  def delete_row(self, index):
    assert index < self.n_rows()
    for column in list(self._columns.values()):
      del column[index]

  def __copy__(self):
    new = loop()
    new._columns = self._columns.copy()
    new.keys_lower = self.keys_lower.copy()
    return new

  copy = __copy__

  def __deepcopy__(self, memo):
    new = loop()
    new._columns = copy.deepcopy(self._columns, memo)
    new.keys_lower = copy.deepcopy(self.keys_lower, memo)
    return new

  def deepcopy(self):
    return copy.deepcopy(self)

  def show(self, out=None, indent="  ", indent_row=None, fmt_str=None, align_columns=True):
    assert self.n_rows() > 0 and self.n_columns() > 0
    if out is None:
      out = sys.stdout
    if indent_row is None:
      indent_row = indent
    assert indent.strip() == ""
    assert indent_row.strip() == ""
    print("loop_", file=out)
    for k in list(self.keys()):
      print(indent + k, file=out)
    values = list(self._columns.values())
    if fmt_str is not None:
      # Pretty printing:
      #   The user is responsible for providing a valid format string.
      #   Values are not quoted - it is the user's responsibility to place
      #   appropriate quotes in the format string if a particular value may
      #   contain spaces.
      values = copy.deepcopy(values)
      for i, v in enumerate(values):
        for flex_numeric_type in (flex.int, flex.double):
          if not isinstance(v, flex_numeric_type):
            try:
              values[i] = flex_numeric_type(v)
            except ValueError:
              continue
            else:
              break
      if fmt_str is None:
        fmt_str = indent_row + ' '.join(["%s"]*len(values))
      for i in range(self.size()):
        print(fmt_str % tuple([values[j][i] for j in range(len(values))]), file=out)
    elif align_columns:
      fmt_str = []
      for i, (k, v) in enumerate(self.items()):
        for i_v in range(v.size()):
          v[i_v] = format_value(v[i_v])
        # exclude and semicolon text fields from column width calculation
        v_ = flex.std_string(item for item in v if "\n" not in item)
        width = v_.max_element_length()
        # See if column contains only number, '.' or '?'
        # right-align numerical columns, left-align everything else
        v = v.select(~( (v == ".") | (v == "?") ))
        try:
          flex.double(v)
        except ValueError:
          width *= -1
        fmt_str.append("%%%is" %width)
      fmt_str = indent_row + "  ".join(fmt_str)
      for i in range(self.size()):
        print((fmt_str %
                       tuple([values[j][i]
                              for j in range(len(values))])).rstrip(), file=out)
    else:
      for i in range(self.size()):
        values_to_print = [format_value(values[j][i]) for j in range(len(values))]
        print(' '.join([indent] + values_to_print), file=out)

  def __str__(self):
    s = StringIO()
    self.show(out=s)
    return s.getvalue()

  def iterrows(self):
    keys = list(self.keys())
    for j in range(self.size()):
      yield OrderedDict(list(zip(keys, [list(self.values())[i][j] for i in range(len(self))])))

  def sort(self, key=None, reverse=False):
    self._columns = OrderedDict(
      sorted(list(self._columns.items()), key=key, reverse=reverse))

  def __eq__(self, other):
    if (len(self) != len(other) or
        self.size() != other.size() or
        list(self.keys()) != list(other.keys())):
      return False
    for value, other_value in zip(list(self.values()), list(other.values())):
      if (value == other_value).count(True) != len(value):
        return False
    return True


def common_substring(seq):
  # DDL1 dictionaries permit a cif loop to contain a local prefix as the
  # first element of any data name:
  #
  #   http://www.iucr.org/resources/cif/spec/ancillary/reserved-prefixes

  if len(seq) == 1: return seq[0]
  substr = seq[0]
  for s in seq:
    substr = LCSubstr_set(substr, s).pop()
  if not (substr.endswith('_') or substr.endswith('.')):
    if '.' in substr:
      substr = substr.split('.')[0]
    elif substr not in seq and substr.count('_') > 1:
      substr = '_'.join(substr.split('_')[:-1])
  return substr

def LCSubstr_set(S, T):
  """Longest common substring function taken from:
  http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python"""

  m = len(S); n = len(T)
  L = [[0] * (n+1) for i in range(m+1)]
  LCS = set()
  longest = 0
  for i in range(m):
    for j in range(n):
      if S[i] == T[j]:
        v = L[i][j] + 1
        L[i+1][j+1] = v
        if v > longest:
          longest = v
          LCS = set()
        if v == longest:
          LCS.add(S[i-v+1:i+1])
  return LCS


import re
_ordinary_char_set = r"!%&()*+,\-./0-9:<=>?@A-Z\\^`a-z{|}~"
_non_blank_char_set = r"%s\"#$'_;[\]"%_ordinary_char_set
_any_print_char_set = r"%s\"#$'_ \t;[\]" %_ordinary_char_set
_text_lead_char_set = r"%s\"#$'_ \t[\]" %_ordinary_char_set

any_print_char_re = re.compile(r"[%s]*" %_any_print_char_set)
tag_re = re.compile(r"_[%s]+" %_non_blank_char_set)
unquoted_string_re = re.compile(
  r"[%s][%s]*" %(_ordinary_char_set, _non_blank_char_set))
quoted_string_re = re.compile(r"('|\")[%s]*('|\")" %_any_print_char_set)
semicolon_string_re = re.compile(r"(\s*)(;).*?(;)(\s*)", re.DOTALL)

def format_value(value_string):
  string_is_quoted = False
  if not value_string:
    return "''"
  # s[0] == "a" is appears to be faster than s.startswith("a")
  if value_string[0] in ("'",'"'):
    m = re.match(quoted_string_re, value_string)
    string_is_quoted = m is not None
  if not string_is_quoted:
    if len(value_string) == 0:
      return "''"
    elif (';' in value_string and value_string.lstrip()[0] == ';'
          and  re.match(semicolon_string_re, value_string) is not None):
      # a semicolon text field
      return "\n%s\n" %value_string.strip()
    elif '\n' in value_string:
      if value_string[0] != '\n':
        value_string = '\n' + value_string
      if value_string[-1] != '\n':
        value_string = value_string + '\n'
      return "\n;%s;\n" %value_string
    elif (value_string[0] in ('#','$','[',']','_')):
      return "'%s'" %value_string
    else:
      for ws in string.whitespace:
        if ws in value_string:
          if ("'" + ws) in value_string:
            if ('"' + ws) in value_string:
              # string can't be quoted, use semi-colon text field instead
              return "\n;\n%s\n;\n" %value_string
            return '"%s"' %value_string
          return "'%s'" %value_string
  return value_string
