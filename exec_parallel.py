
import argparse
import subprocess

import random
import time
import json
from os.path import join, abspath, exists
from os import makedirs
import pprint
import sys
import traceback
import yaml
import copy

import commands
from utils.sync_jobs import JobManager
from utils.filelock import FileLock, Timeout

pp = pprint.PrettyPrinter(indent=1)

def _check_job_running(jobman, sync_dir, name):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    with open(coordinate_path, 'r+') as f:
        body = f.read()
        lis = json.loads(body)
        for i, (job_name, status, job_id, dic) in enumerate(lis):
            if job_name == name:
                return jobman.check_alive(job_id)
    return False

def _init_coordinate(sync_dir):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    lock_path = abspath(join(sync_dir, 'coordinate.json.lock'))
    lock = FileLock(lock_path, timeout=1)

    with lock:
        if not exists(coordinate_path):
            with open(coordinate_path, "w") as f:
                f.write('[]')
                return None, None

def _next_job(jobman, sync_dir, my_id):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    lock_path = abspath(join(sync_dir, 'coordinate.json.lock'))
    lock = FileLock(lock_path, timeout=1)

    with lock:
        with open(coordinate_path, 'r+') as f:
            body = f.read()
            lis = json.loads(body)
            #print('\n[ExecParallel] All hyperparams:')
            #pp.pprint(lis)
            #print('')
            for i, (job_name, status, job_id, dic) in enumerate(lis):
                if status == "not completed":
                    if not jobman.check_alive(job_id):
                        lis[i] = (job_name, status, my_id, dic)
                        f.seek(0)
                        f.write(json.dumps(lis))
                        f.truncate()
                        return job_name, dic
                elif status == "completed":
                    pass
                else:
                    raise NotImplementedError
    return None, None

def _update_job(sync_dir, job_name, status, job_id, dic):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    assert exists(coordinate_path)

    lock_path = abspath(join(sync_dir, 'coordinate.json.lock'))
    lock = FileLock(lock_path, timeout=1)
    with lock:
        with open(coordinate_path, 'r+') as f:
            lis = json.loads(f.read())
            for i, (name, _, _, _) in enumerate(lis):
                if name == job_name:
                    lis[i] = (job_name, status, job_id, dic)
                    f.seek(0)
                    f.write(json.dumps(lis))
                    f.truncate()
                    return
            lis.append((job_name, status, job_id, dic))
            f.seek(0)
            f.write(json.dumps(lis))
            f.truncate()

def _count_jobs(sync_dir):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    assert exists(coordinate_path)

    with open(coordinate_path, 'r') as f:
        lis = json.loads(f.read())
        return len(lis)

def exec_parallel(exp_name, cmd_name, cfg, gpu_id, prefix=""):
    force_ignore_sync = cfg['force_ignore_sync']
    sync_dir = cfg['sync_dir']
    sync_dir = abspath(join(sync_dir, exp_name))
    if not exists(sync_dir):
        try:
            makedirs(sync_dir)
        except Exception as e:
            print('[ExecParallel] Caught Exception:', e.args)

    now = time.time()
    job_id = str(now)
    search_space = cfg['hparams_grid']
    assert search_space is not None

    _init_coordinate(sync_dir)
    jobman = JobManager(sync_dir, job_id, sync_script_path="utils/sync_jobs.py")
    jobman.clear()
    jobman.start()

    completed_dics = []
    errors = []
    job_counter = 0
    no_job_counter = 0
    parallel_counter = 0
    max_jobs = 1
    max_no_job = 100
    max_parallel = 1
    for k, cands in search_space.items():
        max_jobs *= len(cands)

    assert (cmd_name in cfg['allowed_commands']) or (cfg['allowed_commands'] is None)
    cmd = getattr(getattr(commands, cmd_name), cmd_name)

    while True:
        if job_counter >= max_jobs:
            print("[ExecParallel] Stop searching because all patterns are executed")
            break
        if no_job_counter >= max_no_job:
            if parallel_counter >= max_parallel:
                print("[ExecParallel] Finish grid search because any new pattern is not found.")
                break
            else:
                print("[ExecParallel] Pauses grid search because searched for enough time.")
                print("[ExecParallel] Wait for 6 minutes...")
                time.sleep(60 * 6)

            no_job_counter = 0
            parallel_counter += 1

        # resume to train existing hyperparameters
        """ # deprecated
        while True:
            resume_job_name, dic = _next_job(jobman, sync_dir, job_id)
            if resume_job_name is not None:
                for k in dic:
                    cfg[k] = dic[k]
                print('[ExecParallel] Resume job:', resume_job_name)
                try:
                    cmd(exp_name, cfg, gpu_id, prefix=prefix+resume_job_name)
                    _update_job(sync_dir, resume_job_name, 'completed', job_id, dic)

                    completed_dics.append(dic.copy())
                    job_counter += 1
                    no_job_counter = 0
                    parallel_counter = 0
                except:
                    time.sleep(10)
            else:
                break
        """

        # search for new hyperparameters
        random.seed()
        sampled_dic = {}
        for k, cands in search_space.items():
            sampled_dic[k] = random.sample(cands,k=1)[0]

        # train with new hyperparameters
        job_name = ""
        for k, v in sampled_dic.items():
            job_name += f"{k}_{v}--"
            cfg[k] = v

        if _check_job_running(jobman, sync_dir, job_name) and (not force_ignore_sync):
            no_job_counter += 1
        elif sampled_dic not in completed_dics:
            _update_job(sync_dir, job_name, 'not completed', job_id, sampled_dic)
            print('[ExecParallel] Start job:', job_name)
            try:
                cmd(exp_name, copy.deepcopy(cfg), gpu_id, prefix=prefix+job_name)
                _update_job(sync_dir, job_name, 'completed', job_id, sampled_dic)
            except KeyboardInterrupt as e:
                raise e
            except:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                errors.append((job_name, exc_info))

            completed_dics.append(sampled_dic.copy())
            job_counter += 1
            no_job_counter = 0
            parallel_counter = 0
        else:
            no_job_counter += 1

    if len(errors) > 0:
        print('\nErrors occurred:')
        pp.pprint(errors)
    jobman.stop()

if __name__ == '__main__':
    def load_configs(config):
        with open(config, 'r') as f:
            yml = f.read()
            dic = yaml.load(yml, Loader=yaml.FullLoader)
        return dic

    parser = argparse.ArgumentParser()
    parser.add_argument('cmd_name', type=str, help='specify command name from commands/ dir')
    parser.add_argument('exp_name', type=str, help='specify an experiment name in YAML config')
    parser.add_argument('--config', type=str, default='config.yaml', help='file path for YAML configure file')
    parser.add_argument('--gpu_id', type=int, default=0, help='specify gpu id')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--force_restart', action='store_true')
    parser.add_argument('--force_ignore_sync', action='store_true')

    args = parser.parse_args()

    options = []
    cfgs = load_configs(args.config)
    cfg = cfgs[args.exp_name]
    for k in cfg:
        options.append(f'--{k}')
        options.append(cfg[k])
    cfg['data_parallel'] = (cfg['num_gpus'] > 1)
    cfg['force_restart'] = args.force_restart
    cfg['force_ignore_sync'] = args.force_ignore_sync
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    cfg['__other_configs__'] = cfgs

    exec_parallel(args.exp_name, args.cmd_name, cfg, args.gpu_id)

