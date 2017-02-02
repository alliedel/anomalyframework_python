import logging
import numpy as np
import pickle
import sys
import errno
import os


class dotdictify(dict):
    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise(TypeError, 'expected dict')

    def __setitem__(self, key, value):
        if key is not None and '.' in key:
            myKey, restOfKey = key.split('.', 1)
            target = self.setdefault(myKey, dotdictify())
            if not isinstance(target, dotdictify):
                raise(KeyError, 'cannot set "%s" in "%s" (%s)' % (restOfKey, myKey, repr(target)))
            target[restOfKey] = value
        else:
            if isinstance(value, dict) and not isinstance(value, dotdictify):
                value = dotdictify(value)
            dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        if key is None or '.' not in key:
            return dict.__getitem__(self, key)
        myKey, restOfKey = key.split('.', 1)
        target = dict.__getitem__(self, myKey)
        if not isinstance(target, dotdictify):
            raise(KeyError, 'cannot get "%s" in "%s" (%s)' % (restOfKey, myKey, repr(target)))
        return target[restOfKey]

    def __contains__(self, key):
        if key is None or '.' not in key:
            return dict.__contains__(self, key)
        myKey, restOfKey = key.split('.', 1)
        if not dict.__contains__(self, myKey):
            return False
        target = dict.__getitem__(self, myKey)
        if not isinstance(target, dotdictify):
            return False
        return restOfKey in target

    def setdefault(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]

    def get(self, k, d=None):
        if dotdictify.__contains__(self, k):
            return dotdictify.__getitem__(self, k)
        return d

    # __getstate__ and __setstate__ needed for pickling
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __setattr__ = __setitem__
    __getattr__ = __getitem__


# TODO(allie): Compare parameters dictionaries
# def compare_dot_dicts(dot_dict1, dot_dict2):
#     dot_dict_same =
#     dot_dict_only_me =
#     dot_dict_only_other =
#     for key in default_pars:
#         self.__getitem__(key) == default_pars[key]
#
#     same, only_1, only_2



class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.__dict__ = self


def save_array(arr, filename):
    """
    Save an array to be loaded by python later. Here so we can make universal changes based
    on speed / readability later
    later.
    """
    # pickle.dump(arr, open(filename, 'w'))
    np.save(filename, arr)


def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def open_stdout_logger():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def replace_in_nested_dictionary(dictionary, key_to_replace, new_value):
    num_instances = replace_in_nested_dictionary_recurse(dictionary=dictionary,
                                                         key_to_replace=key_to_replace,
                                                         new_value=new_value)
    if num_instances == 0:
        raise ValueError('{} not found in {}'.format(key_to_replace, dictionary))
    if num_instances > 1:
        raise ValueError('{} found more than once in {}'.format(key_to_replace, dictionary))
    return num_instances


def replace_in_nested_dictionary_recurse(dictionary, key_to_replace, new_value):
    count = 0
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            count += replace_in_nested_dictionary_recurse(value, key_to_replace, new_value)
        elif key == key_to_replace:
            dictionary[key] = new_value
            count += 1
    return count


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
