
import os
import copy
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader

from models.ensemble_evaluater import EnsembleEvaluater
from models.image_classification import ImageClassification
from utils.output_manager import OutputManager
from utils._weight_matching import weight_matching, apply_permutation, set_params_

from utils.learning_transfer import get_learner, get_transferred_params_faster, reset_bn_stats_, update_bn_stats_

def listdic2diclist(list_dic, root=True):
    if len(list_dic) == 0:
        return [dict()]
    keys = list(list_dic.keys())
    if root:
        keys.reverse()
        list_dic = {k: list_dic[k] for k in keys}
    key = keys[0]
    lis = copy.deepcopy(list_dic[key])
    del list_dic[key]
    rest_lis = listdic2diclist(list_dic, root=False)
    ret = []
    for rest in rest_lis:
        for item in lis:
            rest[key] = item
            ret.append(copy.deepcopy(rest))
    return ret

def hparams2prefix(hparams):
    ret = ''
    for key, val in hparams.items():
        ret += f'{key}_{val}--'
    return ret

def load_cfg(cfgs, exp_name, data_parallel=False, output_dir=None, force_restart=False):
    cfgs = copy.deepcopy(cfgs)
    cfg = cfgs[exp_name]
    cfg['__other_configs__'] = cfgs
    cfg['data_parallel'] = data_parallel
    cfg['force_restart'] = force_restart
    if output_dir is not None:
        cfg['output_dir'] = output_dir
    return cfg

def transfer(exp_name, cfg, gpu_id, prefix=''):
    assert cfg['seed'] is None

    outman = OutputManager(cfg['output_dir'], exp_name, cfg['output_prefix_hashing'])
    cfgs = cfg['__other_configs__']

    if cfg['output_prefix_hashing']:
        outman.print(f'Prefix: {outman.preprocess_prefix(prefix)}', prefix=prefix)

    tg_exp_name = cfg['target_exp_name']
    tg_hparams_dic = cfg['target_hparams_grid']
    tg_hparams_list = listdic2diclist(tg_hparams_dic)
    if cfg['source_target_seeds'] is not None:
        for i in range(len(tg_hparams_list)):
            assert tg_hparams_list[i]['seed'] is None
            tg_hparams_list[i]['seed'] = cfg['source_target_seeds'][1]
    if cfg['source_target_seeds_pretrained'] is not None:
        for i in range(len(tg_hparams_list)):
            assert tg_hparams_list[i]['seed_checkpoint'] is None
            tg_hparams_list[i]['seed_checkpoint'] = cfg['source_target_seeds_pretrained'][1]

    tg_cfg = load_cfg(cfgs, tg_exp_name, cfg['data_parallel'], cfg['output_dir'], False)
    tg_outmans = [OutputManager(cfg['output_dir'], tg_exp_name, tg_cfg['output_prefix_hashing']) for _ in tg_hparams_list]

    targets_dict = dict(
        params_list=[],
        inits_list=[],
        hparams_list=[],
        outmans_list=[],
    )
    for (tg_hparams, bo) in zip(tg_hparams_list, tg_outmans):
        (final_learner, init_learner, _, _) = get_learner(tg_exp_name, tg_cfg,
                                                        tg_hparams,
                                                        gpu_id, bo, epoch=-1,
                                                        skip_train=True,
                                                        skip_test=True)
        targets_dict['params_list'].append(dict(list(final_learner.model.named_parameters())))
        targets_dict['inits_list'].append(dict(list(init_learner.model.named_parameters())))
        targets_dict['hparams_list'].append(tg_hparams)
        targets_dict['outmans_list'].append(OutputManager(cfg['output_dir'], tg_exp_name, tg_cfg['output_prefix_hashing']))

    device = init_learner.device

    # 0. Load the source model
    sc_exp_name = cfg['source_exp_name']
    sc_hparams = cfg['source_hparams']
    if cfg['source_target_seeds'] is not None:
        assert sc_hparams['seed'] is None
        sc_hparams['seed'] = cfg['source_target_seeds'][0]
    if cfg['source_target_seeds_pretrained'] is not None:
        assert sc_hparams['seed_checkpoint'] is None
        sc_hparams['seed_checkpoint'] = cfg['source_target_seeds_pretrained'][0]
    sc_cfg = load_cfg(cfgs, sc_exp_name, cfg['data_parallel'], cfg['output_dir'], False)
    sc_outman = OutputManager(cfg['output_dir'], sc_exp_name, sc_cfg['output_prefix_hashing'])
    (sc_learner,sc_learner_init,_,sc_cfg_with_hparams) = get_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, epoch=-1, skip_train=True, skip_test=True)
    sc_param_init = dict(list(sc_learner_init.model.named_parameters()))
    sc_param = dict(list(sc_learner.model.named_parameters()))

    # 2. Transfer to (multiple) target models
    import utils._weight_matching
    get_permutation_spec = getattr(utils._weight_matching, sc_cfg_with_hparams['model.config_name'] + '_permutation_spec')
    ps = get_permutation_spec(sc_cfg_with_hparams['bn_affine'])

    for (tg_param, tg_param_init, tg_hparams, tg_outman) in zip(targets_dict['params_list'],
                                        targets_dict['inits_list'],
                                        targets_dict['hparams_list'],
                                        targets_dict['outmans_list']):
        tg_cfg_copy = copy.deepcopy(tg_cfg)
        for key, val in tg_hparams.items():
            tg_cfg_copy[key] = val
        tg_cfg_copy['epoch'] = cfg['epoch_ft']
        tg_cfg_copy['lr'] = cfg['lr_ft']
        if cfg['weight_decay_ft'] is not None:
            tg_cfg_copy['weight_decay'] = cfg['weight_decay_ft']
        if cfg['optimizer_ft'] != -1:
            tg_cfg_copy['optimizer'] = cfg['optimizer_ft']
        if cfg['lr_scheduler_ft'] != -1:
            tg_cfg_copy['lr_scheduler'] = cfg['lr_scheduler_ft']

        if tg_cfg_copy['learning_framework'] == 'ImageClassification':
            trf_learner = ImageClassification(outman, tg_cfg_copy, device, cfg['data_parallel'], silent=True)
        else:
            raise NotImplementedError

        trf_ckp_path = outman.get_abspath(prefix, ext='pth', name='params')
        if os.path.exists(trf_ckp_path):
            trf_ckp = outman.load_checkpoint(prefix=prefix, ext='pth', name='params')
            trf_param = trf_ckp.tr_param
            best_param = trf_ckp.best_param
            best_acc = trf_ckp.best_acc
            outman.print(f'Loaded transferred params. (Best acc: {best_acc})', prefix=prefix)
        else:
            trf_param, best_param = get_transferred_params_faster(ps, sc_exp_name, sc_cfg, sc_hparams, sc_outman, gpu_id,
                                               sc_param_init, sc_param, tg_param_init,
                                               checkpoint_epochs=cfg['checkpoint_epochs'],
                                               tg_exp_name=tg_exp_name,
                                               tg_cfg=copy.deepcopy(tg_cfg),
                                               tg_hparams=tg_hparams,
                                               tg_outman=tg_outman,
                                               outman=outman,
                                               prefix=prefix,
                                               max_iters=cfg['gm_iters'],
                                               base_matching_coeff=cfg['base_matching_coeff'],
                                               normalized_matching=cfg['normalized_matching'],
                                               bn_stats_iters=cfg['bn_stats_iters'],
                                               bn_stats_batch=cfg['bn_stats_batch'],
                                               num_splits=cfg['num_splits'],
                                               baseline_no_perm=cfg['baseline_no_perm'],
                                               baseline_use_target=cfg['baseline_use_target'],
                                               split_scheduling_fn=cfg['split_scheduling_fn'],
                                               num_perms=cfg['num_perms'],
                                               fast_perm_optim=cfg['fast_perm_optim'],
                                               wm_epsilon=cfg['wm_epsilon'],
                                               wm_use_prev_perm=cfg['wm_use_prev_perm'],
                                               branch_at_best=cfg['branch_at_best'],
                                               )
        if cfg['use_best_for_ft']:
            set_params_(trf_learner.model, best_param)
        else:
            set_params_(trf_learner.model, trf_param)
        reset_bn_stats_(trf_learner.model)
        update_bn_stats_(trf_learner, cfg['bn_stats_iters'], cfg['bn_stats_batch'])

        ft_results_path = outman.get_abspath(prefix, ext='json', name='ft_results')
        if os.path.exists(ft_results_path):
            outman.print('FT results are saved at:', str(ft_results_path), prefix=prefix)
        else:
            # Fine-tuning param3
            total_iters = 0
            train_accs, val_accs, losses, indices = [], [], [], []
            for ep in range(cfg['epoch_ft']):
                outman.print('[', str(datetime.datetime.now()) , '] Epoch: ', str(ep), prefix=prefix)
                # Train
                results_train = trf_learner.train(ep, total_iters)
                train_accuracy = results_train['moving_accuracy']
                results_per_iter = results_train['per_iteration']
                new_total_iters = results_train['iterations']
                outman.print('Train Accuracy:', str(train_accuracy), prefix=prefix)
                # Evaluate
                results_eval = trf_learner.evaluate()
                val_accuracy = results_eval['accuracy']
                outman.print('Val Accuracy:', str(val_accuracy), prefix=prefix)
                train_accs.append(train_accuracy)
                val_accs.append(val_accuracy)
                losses += [res['mean_loss'] for res in results_per_iter]
                indices += list(range(total_iters, new_total_iters))
                total_iters = new_total_iters

            results_test = trf_learner.evaluate(dataset_type='test')
            test_accuracy = results_test['accuracy']
            outman.print('Test Accuracy:', str(test_accuracy), prefix=prefix)

            ft_results = {
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'test_acc': test_accuracy,
                    }
            ft_results_path = outman.save_json('ft_results', ft_results, prefix=prefix)
            outman.print('Saved ft results at:', ft_results_path, prefix=prefix)

            model_ckp = outman.new_checkpoint(
                    model_state_dict=trf_learner.model.state_dict(),
                    )
            ft_model_path = outman.save_checkpoint(model_ckp, prefix=prefix, ext='pth', name='ft_model')
            outman.print('Saved ft model at:', ft_model_path, prefix=prefix)

    outman.print('', prefix=prefix)

