
import matplotlib.pyplot as plt
import torch
import copy
import os
import numpy as np
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader

from utils.output_manager import OutputManager
from utils._weight_matching import weight_matching, apply_permutation, set_params_
from utils.learning_transfer import get_learner, get_mid_learner, reset_bn_stats_, update_bn_stats_
import utils._weight_matching

from utils.weight_matching import copy_named_parameters, WeightMatching

def load_cfg(cfgs, exp_name, data_parallel=False, output_dir=None, force_restart=False):
    cfgs = copy.deepcopy(cfgs)
    cfg = cfgs[exp_name]
    cfg['__other_configs__'] = cfgs
    cfg['data_parallel'] = data_parallel
    cfg['force_restart'] = force_restart
    if output_dir is not None:
        cfg['output_dir'] = output_dir
    return cfg

def synced_transfer(exp_name, cfg, gpu_id, prefix=''):
    # Preparation
    cfgs = cfg['__other_configs__']
    outman = OutputManager(cfg['output_dir'], exp_name, cfg['output_prefix_hashing'])
    if cfg['output_prefix_hashing']:
        outman.print(f'Prefix: {outman.preprocess_prefix(prefix)}', prefix=prefix)

    sc_exp_name = cfg['source_exp_name']
    sc_hparams = cfg['source_hparams']
    sc_cfg = load_cfg(cfgs, sc_exp_name, cfg['data_parallel'], cfg['output_dir'], False)
    sc_outman = OutputManager(cfg['output_dir'], sc_exp_name, sc_cfg['output_prefix_hashing'])

    tg_exp_name = cfg['target_exp_name']
    tg_hparams = cfg['target_hparams']
    tg_cfg = load_cfg(cfgs, tg_exp_name, cfg['data_parallel'], cfg['output_dir'], False)
    tg_outman = OutputManager(cfg['output_dir'], tg_exp_name, tg_cfg['output_prefix_hashing'])

    _, sc_learner, _, sc_cfg_with_hparams = get_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, epoch=-1, skip_test=True)
    _, gm_learner, _, _                   = get_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, epoch=-1, skip_test=True)
    _, tg_learner, _, _                   = get_learner(tg_exp_name, tg_cfg, tg_hparams, gpu_id, tg_outman, epoch=-1, skip_test=True)
    _, tr_learner, _, _                   = get_learner(tg_exp_name, tg_cfg, tg_hparams, gpu_id, tg_outman, epoch=-1, skip_test=True)
    device = sc_learner.device

    sc_param_init = {k: p.detach().clone() for k, p in sc_learner.model.named_parameters()}
    tg_param_init = {k: p.detach().clone() for k, p in tg_learner.model.named_parameters()}
    tr_param = {k: p.detach().clone() for k, p in tr_learner.model.named_parameters()}
    assert sum([(tr_param[k] - tg_param_init[k]).abs().sum().item() for k in tr_param]) == 0.0

    if cfg['broken_invariance']:
        get_permutation_spec = getattr(utils._weight_matching, sc_cfg_with_hparams['model.config_name'] + '_permutation_spec_noinv')
    else:
        get_permutation_spec = getattr(utils._weight_matching, sc_cfg_with_hparams['model.config_name'] + '_permutation_spec')
    ps = get_permutation_spec(sc_cfg_with_hparams['bn_affine'])
    wm = WeightMatching(ps, epsilon=cfg['wm_epsilon'], device=device)
    perm = None

    use_random_perm = cfg['use_random_perm']
    if use_random_perm:
        perm_sizes = {p: sc_param_init[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
        perm = {p: np.arange(n) for p, n in perm_sizes.items()}
        for p in perm:
            np.random.shuffle(perm[p])
        perm = {p: torch.LongTensor(perm[p]).to(device) for p in perm}

    use_true_grads = cfg['use_true_grads']
    if use_true_grads:
        true_wm = WeightMatching(ps, epsilon=cfg['wm_epsilon'], device=device)

    batch_size = sc_learner.cfg['batch_size']
    num_workers = sc_learner.cfg['num_workers']
    gm_iters = cfg['gm_iters']
    assert gm_iters == 1 # to use same mini-batch in training & grad matching
    sc_checkpoints, batch_checkpoints = [], []
    sc_val_accs, tg_val_accs, tr_val_accs = [], [], []
    for ep in range(cfg['epoch']):
        # 1. Train for 1 epoch
        step_before_train = hasattr(sc_learner.scheduler, "step_before_train") and sc_learner.scheduler.step_before_train
        if step_before_train:
            try:
                sc_learner.scheduler.step(epoch=ep)
                tg_learner.scheduler.step(epoch=ep)
            except:
                sc_learner.scheduler.step()
                tg_learner.scheduler.step()
        dataloader = DataLoader(sc_learner.train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)
        wm_interval = 1 + int(len(dataloader) / cfg['num_splits'])
        for _it, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if (not use_true_grads) and (not use_random_perm) \
                    and ((cfg['wm_max_iter'] is None) or (ep*int(len(dataloader) / wm_interval) + int(_it / wm_interval) < cfg['wm_max_iter'])) \
                    and ((cfg['wm_max_epoch'] is None) or (ep < cfg['wm_max_epoch'])) \
                    and (_it % wm_interval == 0):
                print(f'debug: Compute permutation @ _it = {_it}')
                sc_param_ckp = {k: p.clone().detach() for k, p in sc_learner.model.named_parameters()}
                if cfg['fast_perm_optim']:
                    sc_checkpoints = [ sc_param_ckp ]
                    batch_checkpoints = [ (inputs, targets) ]
                else:
                    sc_checkpoints.append( sc_param_ckp )
                    batch_checkpoints.append( (inputs, targets) )
                    wm = WeightMatching(ps, epsilon=cfg['wm_epsilon'], device=device)
                for ckp_param, (in_inputs, in_targets) in zip(sc_checkpoints, batch_checkpoints):
                    diff = {k: ckp_param[k] - sc_param_init[k] for k in ckp_param}
                    diff_perm = apply_permutation(ps, perm, diff, device=device)
                    tr_param = {k: tg_param_init[k] + diff_perm[k] for k in diff_perm}

                    set_params_(gm_learner.model, ckp_param)
                    set_params_(tr_learner.model, tr_param)
                    gm_learner.model.eval()
                    tr_learner.model.eval()

                    gm_learner.model.zero_grad()
                    tr_learner.model.zero_grad()
                    sc_outputs = gm_learner.model(in_inputs)
                    tr_outputs = tr_learner.model(in_inputs)
                    sc_loss = gm_learner.criterion(sc_outputs, in_targets)
                    tr_loss = tr_learner.criterion(tr_outputs, in_targets)
                    sc_loss.backward()
                    tr_loss.backward()

                    sc_grad = {k: p.grad.detach().clone() for k, p in gm_learner.model.named_parameters()}
                    tr_grad = {k: p.grad.detach().clone() for k, p in tr_learner.model.named_parameters()}
                    wm.add(tr_grad, sc_grad)

                perm = wm.solve(normalize=cfg['normalized_matching'],
                                silent=True,
                                init_perm=perm if cfg['wm_use_prev_perm'] else None)
                if not cfg['fast_perm_optim']:
                    del wm

            sc_learner.model.train()
            tg_learner.model.train()
            sc_learner.optimizer.zero_grad()
            tg_learner.optimizer.zero_grad()
            sc_outputs = sc_learner.model(inputs)
            tg_outputs = tg_learner.model(inputs)
            sc_loss = sc_learner.criterion(sc_outputs, targets)
            tg_loss = tg_learner.criterion(tg_outputs, targets)
            sc_loss.backward()
            tg_loss.backward()
            sc_learner.optimizer.step()
            tg_learner.optimizer.step()

            if use_true_grads \
                    and ((cfg['wm_max_epoch'] is None) or (ep < cfg['wm_max_epoch'])) \
                    and ((cfg['wm_max_iter'] is None) or (ep*int(len(dataloader) / wm_interval) + int(_it / wm_interval) < cfg['wm_max_iter'])) \
                    and (_it % wm_interval == 0):
                print(f'debug: Compute permutation @ _it = {_it}')
                sc_grad = {k: p.grad.detach().clone() for k, p in sc_learner.model.named_parameters()}
                tg_grad = {k: p.grad.detach().clone() for k, p in tg_learner.model.named_parameters()}
                true_wm.add(tg_grad, sc_grad)
                perm = true_wm.solve(normalize=cfg['normalized_matching'],
                                     silent=True,
                                     init_perm=perm if cfg['wm_use_prev_perm'] else None)

        if not step_before_train:
            try:
                sc_learner.scheduler.step(epoch=ep)
                tg_learner.scheduler.step(epoch=ep)
            except:
                sc_learner.scheduler.step()
                tg_learner.scheduler.step()
        sc_learner.model.eval()
        tg_learner.model.eval()
        sc_val = sc_learner.evaluate()
        tg_val = tg_learner.evaluate()
        outman.print(f"[{ep}] Accuracy:  {sc_val['accuracy']} (source),  {tg_val['accuracy']} (target)", prefix=prefix)

        # 3. Transfer source trajectory
        sc_diff = {k: p - sc_param_init[k] for k, p in sc_learner.model.named_parameters()}
        sc_diff_perm = apply_permutation(ps, perm, sc_diff, device=device)
        tr_param = {k: tg_param_init[k] + sc_diff_perm[k] for k in sc_diff_perm}
        set_params_(tr_learner.model, tr_param)
        tr_learner.model.eval()
        tr_val = tr_learner.evaluate()
        outman.print(f"           {tr_val['accuracy']} (transferred)", prefix=prefix)

        sc_val_accs.append(sc_val['accuracy'])
        tg_val_accs.append(tg_val['accuracy'])
        tr_val_accs.append(tr_val['accuracy'])

    sc_param = {k: p.detach().clone() for k, p in sc_learner.model.named_parameters()}
    tg_param = {k: p.detach().clone() for k, p in tg_learner.model.named_parameters()}

    ckp = outman.new_checkpoint(
        sc_param=sc_param,
        tg_param=tg_param,
        tr_param=tr_param,
        sc_val_accs=sc_val_accs,
        tg_val_accs=tg_val_accs,
        tr_val_accs=tr_val_accs,
    )
    ckp_path = outman.save_checkpoint(ckp, prefix=prefix, ext='pth', name='synced_transfer_results')
    outman.print(f'Saved synced-transfer results at: {ckp_path}', prefix=prefix)
