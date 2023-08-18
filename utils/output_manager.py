import os
import torch
import pprint
import json
import hashlib

class Checkpoint(object):
    def __init__(self, **kwargs):
        self.__dict__['kv_dic'] = kwargs

    def to_dict(self):
        return self.__dict__['kv_dic']

    def has_key(self, key):
        return key in self.__dict__['kv_dic']

    def add_key(self, key, val=None):
        assert key not in self.__dict__['kv_dic']
        self.__dict__['kv_dic'][key] = val

    def __getattr__(self, key):
        return self.__dict__['kv_dic'][key]

    def __setattr__(self, key, val):
        assert key in self.__dict__['kv_dic']
        self.__dict__['kv_dic'][key] = val

class OutputManager(object):
    def __init__(self, output_dir, name, prefix_hashing=False):
        self.output_dir = output_dir
        self.name = name
        self.save_dir = os.path.join(self.output_dir, name)

        self.prefix_hashing = prefix_hashing
        assert (self.prefix_hashing == True) or (self.prefix_hashing == False)

        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except Exception as e:
                print('[OutputManager] Caught Exception:', e.args)

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except Exception as e:
                print('[OutputManager] Caught Exception:', e.args)

    def preprocess_prefix(self, prefix):
        if self.prefix_hashing:
            hash = hashlib.md5(prefix.encode('utf-8')).hexdigest()
            hash_prefix_path = self._get_abspath(hash, 'prefix')
            if not os.path.exists(hash_prefix_path):
                with open(hash_prefix_path, 'w') as f:
                    f.write(prefix+'\n')
            return hash
        else:
            return prefix

    def new_checkpoint(self, **kwargs):
        return Checkpoint(**kwargs)

    def save_checkpoint(self, ckp, prefix="dump", ext="pth", name=None):
        dic = ckp.to_dict()
        prefix = self.preprocess_prefix(prefix)
        filepath = self._get_abspath(prefix, ext, name)
        with open(filepath, 'wb') as f:
            torch.save(dic, f)
        return filepath

    def save_checkpoint_as_json(self, ckp, prefix="dump", name=None):
        dic = ckp.to_dict()
        prefix = self.preprocess_prefix(prefix)
        filepath = self._get_abspath(prefix, 'json', name)
        with open(filepath, 'w') as f:
            json.dump(dic, f, indent=2)

    def load_checkpoint(self, prefix="dump", ext="pth", name=None):
        prefix = self.preprocess_prefix(prefix)
        filepath = self._get_abspath(prefix, ext, name)
        return self.load_checkpoint_from_path(filepath)

    def load_checkpoint_from_path(self, filepath):
        if not os.path.exists(filepath):
            raise Exception('[OutputManager] File not found:', filepath)
        return Checkpoint(**torch.load(filepath))

    def save_json(self, name, dic, prefix="dump"):
        prefix = self.preprocess_prefix(prefix)
        filepath = self._get_abspath(prefix, 'json', name)
        with open(filepath, 'w') as f:
            json.dump(dic, f, indent=2)
        return filepath

    def load_json(self, name, prefix='dump'):
        prefix = self.preprocess_prefix(prefix)
        filepath = self._get_abspath(prefix, 'json', name)
        with open(filepath, 'r') as f:
            dic = json.loads(f.read())
        return dic

    def get_abspath(self, prefix, ext, name=None):
        prefix = self.preprocess_prefix(prefix)
        return self._get_abspath(prefix, ext, name=name)
    def _get_abspath(self, prefix, ext, name=None):
        if name is None:
            name = self.name
        return os.path.abspath(os.path.join(self.save_dir, f'{prefix}.{name}.{ext}'))

    def add_log(self):
        pass

    def print_filepath(self, prefix=""):
        return os.path.join(self.save_dir, f'{prefix}.{self.name}.out')

    def print(self, *args, prefix=""):
        print(*args)
        prefix = self.preprocess_prefix(prefix)
        print(*args, file=open(self.print_filepath(prefix=prefix), "a+"))

    def pprint(self, *args, prefix=""):
        s = pprint.pformat(*args, indent=1)
        self.print(s, prefix=prefix)

if __name__ == '__main__':
    outman = OutputManager('test', 'outman', prefix_hashing=True)
    outman.print("a", "b", prefix="thisisprefix")
    outman.print("c", "d", prefix="thisisprefix")
