
import copy
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader
from torch.nn import BatchNorm2d
import math

from models.ensemble_evaluater import EnsembleEvaluater
from models.image_classification import ImageClassification
from utils.output_manager import OutputManager
from utils._weight_matching import apply_permutation, set_params_
from utils.weight_matching import copy_named_parameters, WeightMatching

from commands.train import train

def get_learner(exp_name, cfg, hparams, gpu_id, outman, epoch=-1, skip_train=False, skip_test=False):
    cfg = copy.deepcopy(cfg)
    prefix = ''
    for key, value in hparams.items():
        cfg[key] = value
        prefix += f'{key}_{value}--'

    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    if not skip_train:
        results = train(exp_name, cfg, gpu_id, prefix=prefix, skip_test=skip_test)
    else:
        results = None

    if cfg['learning_framework'] == 'ImageClassification':
        learner = ImageClassification(outman, cfg, device, cfg['data_parallel'], silent=True)
        early_learner = ImageClassification(outman, cfg, device, cfg['data_parallel'], silent=True)
    else:
        raise NotImplementedError
    dump_prefix = f'dump.{prefix}'
    ckp = outman.load_checkpoint(prefix=dump_prefix, ext="pth")
    assert ckp is not None
    learner.model.load_state_dict(ckp.model_state_dict)

    if epoch is None:
        early_learner = None
    else:
        mid_ckp = outman.load_checkpoint(prefix=f'epoch{epoch}.{prefix}', ext="pth")
        early_learner.model.load_state_dict(mid_ckp.model_state_dict)

    return (learner, early_learner, results, cfg)

def reset_bn_stats_(model):
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            m.momentum = None
            m.reset_running_stats()

def transfer_bn_params_(ps, perm, sc_model, tg_model, device):
    sc_param = dict(list(sc_model.named_parameters()))
    sc_param_perm = apply_permutation(ps, perm, sc_param, device=device)

    sc_bns = dict()
    tg_bns = dict()
    for name, m in sc_model.named_modules():
        if isinstance(m, BatchNorm2d):
            sc_bns[name] = m
    for name, m in tg_model.named_modules():
        if isinstance(m, BatchNorm2d):
            tg_bns[name] = m
    for name in sc_bns:
        tg_bns[name].weight.data = sc_param_perm[name + '.weight'].data
        tg_bns[name].bias.data = sc_param_perm[name + '.bias'].data

def save_transfer_results(outman, prefix, val_accs, tg_param_init, tr_param, best_acc, best_iter, best_param, perms):
    outman.print(f'val_accs: {val_accs}', prefix=prefix)
    results = {
            'val_accs': val_accs,
            }
    results_path = outman.save_json('transfer_results', results, prefix=prefix)
    outman.print('Saved results at:', prefix=prefix)
    outman.print(f'  {results_path}', prefix=prefix)

    # NOTE: not support batch norms
    ckp = outman.new_checkpoint(
        tg_param_init=tg_param_init,
        tr_param=tr_param,
        best_acc=best_acc,
        best_iter=best_iter,
        best_param=best_param,
        perms=perms,
    )
    ckp_name = 'params'
    outman.save_checkpoint(ckp, prefix=prefix, ext='pth', name=ckp_name)
    ckp_path = outman.get_abspath(prefix, ext='pth', name=ckp_name)
    outman.print('Saved params at:', prefix=prefix)
    outman.print(f'  {ckp_path}', prefix=prefix)

def get_transferred_params_faster(ps, sc_exp_name, sc_cfg, sc_hparams, sc_outman, gpu_id,
                           sc_param_init, sc_param, tg_param_init,
                           checkpoint_epochs,
                           tg_exp_name=None,
                           tg_cfg=None,
                           tg_hparams=None,
                           tg_outman=None,
                           outman=None,
                           prefix=None,
                           max_iters=None,
                           base_matching_coeff=None,
                           normalized_matching=None,
                           bn_stats_iters=None,
                           bn_stats_batch=None,
                           num_splits=None,
                           baseline_no_perm=None,
                           baseline_use_target=None,
                           split_scheduling_fn=None,
                           num_perms=None,
                           fast_perm_optim=False,
                           wm_epsilon=None,
                           wm_use_prev_perm=None,
                           branch_at_best=None,
                           ):

    def scheduler(idx, T):
        if split_scheduling_fn is None:
            ret = idx
            Z = T
        else:
            ret = 0.
            Z = 0.
            for t in range(idx):
                ret += eval(split_scheduling_fn)
            for t in range(T):
                Z += eval(split_scheduling_fn)
        #print(f'[debug] learning_transfer.scheduler({idx}, {T}): {ret / Z}')
        return ret / Z

    (sc_learner_for_gm,_,_,sc_cfg_with_hparams) = get_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, None, skip_train=True, skip_test=True)
    (tr_learner,_,_,_) = get_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, None, skip_train=True, skip_test=True)

    perm = None
    fixed_perms = [perm]
    assert num_splits is not None
    if num_splits == -1:
        baseline_no_perm = True
        num_splits = 1

    if type(checkpoint_epochs) is str:
        ckp_epochs = eval(checkpoint_epochs)
    else:
        ckp_epochs = checkpoint_epochs
    assert ckp_epochs[0] == -1
    assert len(ckp_epochs) > 1
    outman.print(f'\nckp_epochs: {ckp_epochs}\n', prefix=prefix)

    cache_mid_learners = dict()
    def get_interp_param(epoch1=None, epoch2=None, coeff=None):
        assert epoch1 is not None

        if epoch1 not in cache_mid_learners:
            cache_mid_learners[epoch1] = get_mid_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, epoch1)
        learner1 = cache_mid_learners[epoch1]
        param1 = dict(list(learner1.model.named_parameters()))
        param1 = {k: p.clone().detach() for k, p in param1.items()}

        if epoch2 is None:
            assert coeff is None
            return param1

        if epoch2 not in cache_mid_learners:
            cache_mid_learners[epoch2] = get_mid_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, epoch2)
        learner2 = cache_mid_learners[epoch2]
        param2 = dict(list(learner2.model.named_parameters()))
        param2 = {k: p.clone().detach() for k, p in param2.items()}

        interp_param = {key: param1[key] + coeff*(param2[key] - param1[key]) for key in param1}
        return interp_param

    ckp_tuple_list = []
    ckp_tuple_list.append({'epoch1': ckp_epochs[0]})
    for i in range(1, len(ckp_epochs)):
        for idx in range(1, num_splits + 1):
            ckp_tuple_list.append({'epoch1': ckp_epochs[i-1], 'epoch2': ckp_epochs[i], 'coeff': scheduler(idx, num_splits)})
    num_checkpoints = len(ckp_tuple_list)

    if baseline_use_target:
        tg_param_list = []
        for ep in ckp_epochs:
            tgl = get_mid_learner(tg_exp_name, tg_cfg, tg_hparams, gpu_id, tg_outman, ep)
            tg_param_list.append({k: p.clone().detach() for k, p in tgl.model.named_parameters()})
            scl = get_mid_learner(sc_exp_name, sc_cfg, sc_hparams, gpu_id, sc_outman, ep)
            cache_mid_learners[ep] = scl
        tg_counter = 0

    if sc_cfg_with_hparams['learning_framework'] == 'ImageClassification':
        ckp_learner = ImageClassification(outman, sc_cfg_with_hparams, sc_learner_for_gm.device, sc_cfg_with_hparams['data_parallel'], silent=True)
    else:
        raise NotImplementedError()

    assert (num_checkpoints - 1) % num_perms == 0
    perm_sharing_interval = int((num_checkpoints - 1) / num_perms)

    tr_param_fixed = tg_param_init
    start_idx = 0
    prev_start_idx = None
    start_param = get_interp_param(**ckp_tuple_list[start_idx])
    l2_loss = None
    val_accs = []
    best_acc = 0.0
    best_iter = None
    best_param = None
    best_perm = None
    for i1 in range(1, num_checkpoints):
        new_perm_flag = ((i1 - 1) % perm_sharing_interval == 0)
        if num_perms == 1:
            assert (i1 == 1) or (not new_perm_flag)

        if i1 > 1 and new_perm_flag:
            assert num_perms > 1
            prev_start_idx = start_idx
            start_idx = i1 - 1
            del start_param
            start_param = get_interp_param(**ckp_tuple_list[start_idx])

            # Reset checkpoint chaches
            cache_epochs = list(cache_mid_learners.keys())
            for key in cache_epochs:
                if ckp_tuple_list[start_idx]['epoch1'] <= key:
                    pass
                else:
                    del cache_mid_learners[key]

        if i1 % 10 == 0 and fast_perm_optim:
            # Reset all checkpoint chaches
            cache_epochs = list(cache_mid_learners.keys())
            for key in cache_epochs:
                del cache_mid_learners[key]

        # Reset wm
        new_wm_flag = (not fast_perm_optim) or new_perm_flag
        if new_wm_flag:
            wm = WeightMatching(ps, epsilon=wm_epsilon, device=sc_learner_for_gm.device)
        else:
            pass

        # Executed every time execpt for the first perm
        if prev_start_idx is not None:
            assert i1 > 1
            assert num_perms > 1
            prev_start_param = get_interp_param(**ckp_tuple_list[prev_start_idx])
            diff = {key: (start_param[key] - prev_start_param[key]).detach().clone() for key in tr_param_fixed}
            del prev_start_param
            if (branch_at_best and new_perm_flag):
                # use the best perm as the last perm
                diff_perm = apply_permutation(ps, best_perm, diff, device=sc_learner_for_gm.device)
            else:
                # use the last perm
                diff_perm = apply_permutation(ps, perm, diff, device=sc_learner_for_gm.device)
            # Update tr_param_fixed & perms
            if new_perm_flag:
                if branch_at_best:
                    del tr_param_fixed
                    tr_param_fixed = best_param
                    fixed_perms.append(best_perm)
                else:
                    tr_param_fixed = {key: tr_param_fixed[key] + diff_perm[key] for key in tr_param_fixed}
                    fixed_perms.append(perm)
                print('[debug] updated tr_param_fixed')
            # add L2 regularization to prev permutation if wm is initialized
            if new_wm_flag:
                wm.add(diff_perm, diff)

        if baseline_use_target and new_perm_flag:  # include (i1 == 1) case
            tg_param1 = tg_param_list[tg_counter]
            tg_param2 = tg_param_list[tg_counter+1]
            tg_diff = {k: tg_param2[k] - tg_param1[k] for k in tg_param1}

            epoch1 = ckp_epochs[tg_counter]
            epoch2 = ckp_epochs[tg_counter+1]
            sc_param1 = {k: p.clone().detach() for k, p in cache_mid_learners[epoch1].model.named_parameters()}
            sc_param2 = {k: p.clone().detach() for k, p in cache_mid_learners[epoch2].model.named_parameters()}
            sc_diff = {k: sc_param2[k] - sc_param1[k] for k in sc_param}

            wm.add(tg_diff, sc_diff)
            perm = wm.solve(silent=True)
            tg_counter += 1
            del wm

        # Initialize tr_param
        set_params_(tr_learner.model, tr_param_fixed)
        tr_param = {key: p.detach() for (key, p) in tr_learner.model.named_parameters()}

        # Compute perm
        for i2 in range(start_idx, i1):
            if fast_perm_optim and (i2 < i1-1):
                pass
            elif baseline_no_perm:
                pass
            elif baseline_use_target:
                pass
            else:
                param1 = get_interp_param(**ckp_tuple_list[i2])
                set_params_(ckp_learner.model, param1)
                diff = {key: (param1[key] - start_param[key]).detach().clone() for key in tr_param_fixed}
                diff_perm = apply_permutation(ps, perm, diff, device=sc_learner_for_gm.device)
                for key in tr_param:
                    tr_param[key].data += diff_perm[key].data

                tr_learner.model.train()
                ckp_learner.model.train()
                reset_bn_stats_(tr_learner.model)
                reset_bn_stats_(ckp_learner.model)
                tr_grad, ckp_grad = get_sync_grads(sc_learner_for_gm,
                                                    tr_learner.model,
                                                    ckp_learner.model,
                                                    max_iters=max_iters,
                                                    bn_stats_iters=bn_stats_iters,
                                                    bn_stats_batch=bn_stats_batch)
                tr_learner.model.eval()
                wm.add(tr_grad, ckp_grad)

                # Reset tr_param to tr_param_fixed
                for key in tr_param:
                    tr_param[key].data -= diff_perm[key].data
                num_error = (tr_param[key].data - tr_param_fixed[key].data).abs().max().item()
                if num_error >= 1e-7:
                    print('[debug] large num_error:', num_error)
                    assert num_error < 1e-6 # admit some numerical error

        if (not baseline_no_perm) and (not baseline_use_target):
            perm = wm.solve(normalize=normalized_matching,
                            silent=True,
                            init_perm=perm if wm_use_prev_perm else None) # TODO: for efficiency
            l2_loss = wm.calc_loss(perm, device=sc_learner_for_gm.device)

        # Re-initialize tr_param
        set_params_(tr_learner.model, tr_param_fixed)
        tr_param = {key: p.detach() for (key, p) in tr_learner.model.named_parameters()}

        # Transfer by new permutation
        param1 = get_interp_param(**ckp_tuple_list[i1])
        diff = {key: (param1[key] - start_param[key]).detach().clone() for key in tr_param}
        del param1
        diff_perm = apply_permutation(ps, perm, diff, device=sc_learner_for_gm.device)
        for key in tr_param:
            tr_param[key].data += diff_perm[key].data
        # Update BN stats
        tr_learner.model.train()
        reset_bn_stats_(tr_learner.model)
        update_bn_stats_(tr_learner, bn_stats_iters, bn_stats_batch)
        tr_learner.model.eval()
        # Evaluate the transferred model
        results_eval = tr_learner.evaluate()
        val_accuracy = results_eval['accuracy']
        val_accs.append(val_accuracy)
        outman.print(f'[Learning Transfer:{i1}/{num_checkpoints-1}] Val Accuracy:', str(val_accuracy), prefix=prefix)
        if l2_loss is not None:
            outman.print(f'                      Matching Loss:', str(l2_loss), prefix=prefix)
        if val_accuracy > best_acc:
            del best_param
            best_acc = val_accuracy
            best_iter = i1
            best_param = {key: tr_param[key].detach().clone() for key in tr_param}
            best_perm = perm

    fixed_perms.append(perm)

    # Finally initialize tr_param
    set_params_(tr_learner.model, tr_param_fixed)
    tr_param = {key: p.detach() for (key, p) in tr_learner.model.named_parameters()}

    param1 = get_interp_param(**ckp_tuple_list[-1])
    param2 = get_interp_param(**ckp_tuple_list[start_idx])
    last_diff = {key: (param1[key] - param2[key]).detach().clone() for key in tr_param}
    del param1, param2
    last_diff_perm = apply_permutation(ps, perm, last_diff, device=sc_learner_for_gm.device)
    for key in tr_param:
        tr_param[key].data += last_diff_perm[key].data

    if not baseline_no_perm:
        tr_learner.model.train()
        reset_bn_stats_(tr_learner.model)
        update_bn_stats_(tr_learner, bn_stats_iters, bn_stats_batch)
        tr_learner.model.eval()

        results_eval = tr_learner.evaluate()
        val_accuracy = results_eval['accuracy']
        outman.print(f'[Learning Transfer] Final Val Accuracy:', str(val_accuracy), prefix=prefix)

    save_transfer_results(outman, prefix, val_accs, tg_param_init, tr_param, best_acc, best_iter, best_param, fixed_perms)

    del tr_learner
    return tr_param, best_param

def get_grads(learner, model, max_iters=1):
    model.zero_grad()
    loss = 0.0

    batch_size = learner.cfg['batch_size']
    num_workers = learner.cfg['num_workers']
    dataloader = DataLoader(learner.train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    for it, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        outputs = model(inputs)
        loss = learner.criterion(outputs, targets)
        loss.backward()

        if it >= max_iters-1:
            break

    grad = {name: param.grad / max_iters for name, param in model.named_parameters()}
    return grad

def update_bn_stats_(learner, iters, batch_size):
    num_workers = learner.cfg['num_workers']
    dataloader = DataLoader(learner.train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    learner.model.train()
    for it, (inputs, targets) in enumerate(dataloader):
        if it >= iters:
            break
        inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        outputs = learner.model(inputs)
    learner.model.eval()

def get_sync_grads(learner, model1, model2, max_iters,
                   bn_stats_iters, bn_stats_batch):
    model1.zero_grad()
    model2.zero_grad()
    loss1 = None
    loss2 = None
    num_workers = learner.cfg['num_workers']

    # 1. Updating BN stats (in a sync way)
    batch_size = bn_stats_batch
    dataloader = DataLoader(learner.train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    for it, (inputs, targets) in enumerate(dataloader):
        if it >= bn_stats_iters:
            break
        inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        model1(inputs)
        model2(inputs)

    # 2. Calculate gradients (in a sync way)
    batch_size = learner.cfg['batch_size']
    dataloader = DataLoader(learner.train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    for it, (inputs, targets) in enumerate(dataloader):
        if it >= max_iters:
            break
        inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        outputs1, outputs2 = model1(inputs), model2(inputs)
        loss1 = learner.criterion(outputs1, targets)
        loss2 = learner.criterion(outputs2, targets)
        loss1.backward()
        loss2.backward()

    grad1 = {name: param.grad / max_iters for name, param in model1.named_parameters()}
    grad2 = {name: param.grad / max_iters for name, param in model2.named_parameters()}

    return (grad1, grad2)

def grad_matching(ps, learner, model1, model2, max_iters=1,
                                               base_matching_coeff=0.0,
                                               consistent_perms=False,
                                               init_param1=None,
                                               init_param2=None,
                                               normalize=False,
                                               ):
    model1.zero_grad()
    model2.zero_grad()
    loss1 = None
    loss2 = None

    batch_size = learner.cfg['batch_size']
    num_workers = learner.cfg['num_workers']
    dataloader = DataLoader(learner.train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    for it, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        outputs1, outputs2 = model1(inputs), model2(inputs)
        if loss1 is None or loss2 is None:
            loss1 = learner.criterion(outputs1, targets)
            loss2 = learner.criterion(outputs2, targets)
        else:
            loss1 += learner.criterion(outputs1, targets)
            loss2 += learner.criterion(outputs2, targets)

        if it >= max_iters-1:
            break

    loss1 /= max_iters
    loss2 /= max_iters
    loss1.backward()
    loss2.backward()

    param1 = {name: param for name, param in model1.named_parameters()}
    param2 = {name: param for name, param in model2.named_parameters()}
    grad1 = {name: param.grad for name, param in model1.named_parameters()}
    grad2 = {name: param.grad for name, param in model2.named_parameters()}
    if consistent_perms:
        assert (init_param1 is not None) and (init_param2 is not None)
        perm = weight_matching(ps,
                           grad1, grad2,
                           params_a_2={k: param1[k] - init_param1[k] for k in param1},
                           params_b_2={k: param2[k] - init_param2[k] for k in param2},
                           coeff=base_matching_coeff,
                           normalize=normalize,
                           silent=True)
    else:
        perm = weight_matching(ps,
                               grad1, grad2,
                               params_a_2=param1,
                               params_b_2=param2,
                               coeff=base_matching_coeff,
                               normalize=normalize,
                               silent=True)
    return perm

def get_mid_learner(exp_name, cfg, hparams, gpu_id, outman, epoch):
    cfg = copy.deepcopy(cfg)
    prefix = ''
    for key, value in hparams.items():
        cfg[key] = value
        prefix += f'{key}_{value}--'

    #results = train(exp_name, cfg, gpu_id, prefix=prefix, skip_test=skip_test)

    learners = []
    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')

    if cfg['learning_framework'] == 'ImageClassification':
        learner = ImageClassification(outman, cfg, device, cfg['data_parallel'], silent=True)
    else:
        raise NotImplementedError

    mid_ckp = outman.load_checkpoint(prefix=f'epoch{epoch}.{prefix}', ext="pth")
    if mid_ckp is None:
        raise Exception(f'Error occurred when loading checkpoint. Maybe not found:', outman.get_abspath(prefix=f'epoch{epoch}.{prefix}', ext="pth"))
    learner.model.load_state_dict(mid_ckp.model_state_dict)

    return learner
