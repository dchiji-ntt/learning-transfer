
import os
import time
import datetime
import copy

import torch
from torch.nn import DataParallel

from commands.test import test
from models.image_classification import ImageClassification
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger

def count_params(model):
    count = 0
    count_not_score = 0
    count_reduced = 0
    for (n,p) in model.named_parameters():
        count += p.flatten().size(0)
        #print(n+':',p.flatten().size(0))
        print(f'{n}:{p.flatten().size(0)} ({p.size()})')
        count_not_score += p.flatten().size(0)
    count_after_pruning = count_not_score - count_reduced
    total_sparsity = 1 - (count_after_pruning / count_not_score)
    print('Params after/before pruned:\t', count_after_pruning, '/', count_not_score, '(sparsity: ' + str(total_sparsity) +')')
    print('Total Params:\t', count)
    return {
            'params_after_pruned': count_after_pruning,
            'params_before_pruned': count_not_score,
            'total_params': count,
            'sparsity': total_sparsity,
            }

def train(exp_name, cfg, gpu_id, prefix="", skip_test=False):
    cfg = copy.deepcopy(cfg)

    if type(cfg['checkpoint_epochs']) is str:
        cfg['checkpoint_epochs'] = eval(cfg['checkpoint_epochs'])
    if cfg['seed'] is not None:
        set_random_seed(cfg['seed'])
    elif cfg['seed_by_time']:
        set_random_seed(int(time.time() * 1000) % 1000000)
    else:
        raise Exception("Set seed value.")
    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    outman = OutputManager(cfg['output_dir'], exp_name, cfg['output_prefix_hashing'])

    dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")

    outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=prefix)

    pd_logger = PDLogger()
    pd_logger.set_filename(outman.get_abspath(prefix=f"pd_log.{prefix}", ext="pickle"))
    if os.path.exists(pd_logger.filename) and not cfg['force_restart']:
        pd_logger.load()

    if cfg['learning_framework'] == 'ImageClassification':
        learner = ImageClassification(outman, cfg, device, cfg['data_parallel'])
    else:
        raise NotImplementedError

    params_info = count_params(learner.model)

    start_epoch = 0
    outman.print(dump_path, prefix=prefix)
    try:
        ckp = outman.load_checkpoint(prefix=f"dump.{prefix}", ext="pth")
    except:
        ckp = None
    if (ckp is not None) and (not cfg['force_restart']):
        start_epoch = ckp.epoch + 1
        if isinstance(learner.model, DataParallel):
            learner.model.module.load_state_dict(ckp.model_state_dict)
        else:
            learner.model.load_state_dict(ckp.model_state_dict)
        learner.optimizer.load_state_dict(ckp.optim_state_dict)
        learner.scheduler.load_state_dict(ckp.sched_state_dict)
    else:
        ckp = outman.new_checkpoint(
                                    epoch=0,
                                    total_iters=0,
                                    best_val=None,
                                    best_epoch=0,
                                    total_seconds=0.,
                                    model_state_dict=None,
                                    optim_state_dict=None,
                                    sched_state_dict=None,
                                   )
        load_checkpoint_flag = (cfg['load_checkpoint_by_hparams'] is not None) or (cfg['load_checkpoint_by_path'] is not None)
        if load_checkpoint_flag:
            if cfg['load_checkpoint_by_hparams'] is not None:
                checkpoint_exp_name = cfg['load_checkpoint_by_hparams']['_exp_name']
                del cfg['load_checkpoint_by_hparams']['_exp_name']
                checkpoint_hparams = cfg['load_checkpoint_by_hparams']
                if cfg['seed_checkpoint'] is not None:
                    checkpoint_hparams['seed'] = cfg['seed_checkpoint']
                checkpoint_outman = OutputManager(cfg['output_dir'], checkpoint_exp_name,
                                                  cfg['__other_configs__'][checkpoint_exp_name]['output_prefix_hashing'])

                checkpoint_job_name = ""
                for k, v in checkpoint_hparams.items():
                    checkpoint_job_name += f"{k}_{v}--"

                model_ckp = checkpoint_outman.load_checkpoint(prefix=f"dump.{checkpoint_job_name}", ext="pth")
            elif cfg['load_checkpoint_by_path'] is not None:
                model_ckp = load_checkpoint_from_path(cfg['load_checkpoint_by_path'])
            else:
                raise NotImplementedError()

            assert model_ckp is not None
            if isinstance(learner.model, DataParallel):
                raise NotImplementedError()
            else:
                if cfg['initialize_head']:
                    learner.model.load_state_dict_without_head(model_ckp.model_state_dict)
                    learner.model.initialize_head(seed=cfg['seed']*2+1)
                else:
                    learner.model.load_state_dict(model_ckp.model_state_dict)
            #learner.optimizer.load_state_dict(checkpoint_dict['optim_state_dict'])
            #if 'sched_state_dict' in checkpoint_dict:
            #    learner.scheduler.load_state_dict(checkpoint_dict['sched_state_dict'])

        if -1 in cfg['checkpoint_epochs']:
            if isinstance(learner.model, DataParallel):
                ckp.model_state_dict = learner.model.module.state_dict()
            else:
                ckp.model_state_dict = learner.model.state_dict()
            ckp.optim_state_dict = learner.optimizer.state_dict()
            ckp.sched_state_dict = learner.scheduler.state_dict()
            outman.save_checkpoint(ckp, prefix=f'epoch-1.{prefix}', ext="pth")
            outman.save_checkpoint(ckp, prefix=f"dump.{prefix}", ext="pth")


    # Training loop
    for _epoch in range(start_epoch, cfg['epoch']):
        ckp.epoch = _epoch
        start_sec = time.time()

        outman.print('[', str(datetime.datetime.now()) , '] Epoch: ', str(ckp.epoch), prefix=prefix)

        # Train
        results_train = learner.train(ckp.epoch, ckp.total_iters)
        train_accuracy = results_train['moving_accuracy']
        results_per_iter = results_train['per_iteration']
        new_total_iters = results_train['iterations']
        total_loss_train = results_train['loss']

        pd_logger.add('train_accs', indices=[ckp.epoch], values=[train_accuracy])
        outman.print('Train Accuracy:', str(train_accuracy), prefix=prefix)
        if cfg['print_train_loss']:
            outman.print('Train Loss:', str(total_loss_train), prefix=prefix)

        # Evaluate
        results_eval = learner.evaluate()
        val_accuracy = results_eval['accuracy']
        pd_logger.add('val_accs', indices=[ckp.epoch], values=[val_accuracy])
        outman.print('Val Accuracy:', str(val_accuracy), prefix=prefix)

        # Save train losses per iteration
        losses = [res['mean_loss'] for res in results_per_iter]
        indices = list(range(ckp.total_iters, new_total_iters))
        pd_logger.add('train_losses', indices=indices, values=losses)
        # Update total_iters
        ckp.total_iters = new_total_iters

        # Flag if save best model
        if (ckp.best_val is None) or (ckp.best_val < val_accuracy):
            ckp.best_val = val_accuracy
            ckp.best_epoch = ckp.epoch
            save_best_model = True
        else:
            save_best_model = False

        end_sec = time.time()
        ckp.total_seconds += end_sec - start_sec

        if isinstance(learner.model, DataParallel):
            ckp.model_state_dict = learner.model.module.state_dict()
        else:
            ckp.model_state_dict = learner.model.state_dict()

        ckp.optim_state_dict = learner.optimizer.state_dict()
        ckp.sched_state_dict = learner.scheduler.state_dict()

        outman.save_checkpoint(ckp, prefix=f"dump.{prefix}", ext="pth")
        if save_best_model and cfg['save_best_model']:
            outman.save_checkpoint(ckp, prefix=f"best.{prefix}", ext="pth")
        if ckp.epoch in cfg['checkpoint_epochs']:
            outman.save_checkpoint(ckp, prefix=f'epoch{ckp.epoch}.{prefix}', ext="pth")

        info_ckp = outman.new_checkpoint(
                                         last_val=val_accuracy,
                                         epoch=ckp.epoch,
                                         best_val=ckp.best_val,
                                         best_epoch=ckp.best_epoch,
                                         loss_train=total_loss_train,
                                         acc_train=train_accuracy,
                                         total_time=str(datetime.timedelta(seconds=int(ckp.total_seconds))),
                                         total_seconds=ckp.total_seconds,
                                         prefix=prefix,
                                         params_info=params_info,
                                        )
        outman.save_checkpoint_as_json(info_ckp, prefix=f"info.{prefix}")

        pd_logger.save()

    if not skip_test:
        return test(exp_name, cfg, gpu_id, prefix=prefix)
    else:
        return None

