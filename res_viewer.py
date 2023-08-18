#!/usr/bin/python3

import argparse
import os
import sys
import yaml
import json
import itertools
import pprint
import hashlib

def get_abspath(exp_dir, exp_name, prefix, ext):
    return os.path.abspath(os.path.join(exp_dir, f'{prefix}.{exp_name}.{ext}'))

if __name__ == '__main__':
    def load_configs(config):
        with open(config, 'r') as f:
            yml = f.read()
            dic = yaml.load(yml, Loader=yaml.FullLoader)
        return dic

    pp = pprint.PrettyPrinter(indent=2)

    help_type = 'specify the result type from: print / ...'
    help_exp_dir = 'specify a result directory for a single experiment'

    if len(sys.argv) < 3:
        print(f'Usage: $ python {sys.argv[0]} <result-type> <exp-name>')
        print(f'    <result-type>: {help_type}')
        print(f'    <exp-dir>: {help_exp_dir}')
        print(f'Options:')
        print(f'    --<hparam-name> <hparam-value>: return filterred results by <hparam-name>=<hparam-value>')
        sys.exit(0)
    exp_dir = sys.argv[2]
    exp_name = exp_dir.split('/')[-1]
    cfgs = load_configs('config.yaml')
    cfg = cfgs[exp_name]

    hash_flag = cfg['output_prefix_hashing']

    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help=help_type)
    parser.add_argument('exp_dir', type=str, help=help_exp_dir)
    for key in cfg['hparams_grid'].keys():
        parser.add_argument(f'--{key}', type=str, default=None)
    args = parser.parse_args()

    assert exp_dir == args.exp_dir

    prefix_list = []
    hparam_names = list(cfg['hparams_grid'].keys())
    for hparams in itertools.product(*[cfg['hparams_grid'][name] for name in hparam_names]):
        prefix = ''
        append_flag = True
        for name, hp in zip(hparam_names, hparams):
            if getattr(args, name) is not None:
                if getattr(args, name) != str(hp):
                    append_flag = False
            prefix += f'{name}_{hp}--'
        if append_flag:
            prefix_list.append(prefix)

    pp.pprint(prefix_list)

    for prefix in prefix_list:
        print(f'=== {prefix} ===')
        if hash_flag:
            prefix = hashlib.md5(prefix.encode('utf-8')).hexdigest()
        if args.type == 'print':
            try:
                with open(get_abspath(exp_dir, exp_name, prefix=prefix, ext='out'), 'r') as f:
                    body = f.read()
                    print(body)
            except:
                print('[res_viewer.py] error occurred during opening file', e)
        elif args.type == 'info':
            try:
                with open(get_abspath(exp_dir, exp_name, prefix=f"info.{prefix}", ext="json"), 'r') as f:
                    info_dic = json.loads(f.read())
                pp.pprint(info_dic)
            except Exception as e:
                print('[res_viewer.py] error occurred during opening file', e)
        elif args.type == 'hash':
            print(f'Hash: {prefix}')
            """ # deprecated
            try:
                pairs = []
                for fname in os.listdir(exp_dir):
                    if len(fname) >= 7 and fname[-7:] == '.prefix':
                        hash = fname[:-7]
                        with open(os.path.join(exp_dir, fname)) as f:
                            pairs.append((hash, f.readlines()[0][:-1]))
                for (h, prefix) in pairs:
                    print('------------')
                    print(h)
                    print(prefix)
            except Exception as e:
                print('[res_viewer.py] error occurred during opening file', e)
            """
        else:
            raise NotImplementedError()


