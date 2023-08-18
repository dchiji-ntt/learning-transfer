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

from models.image_classification import ImageClassification

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

def plot_1dlandscape(pdf_path, learner, param_pairs, labels, colors, markers, linestyles,
                     num_plots=100,
                     linewidth=1.5,
                     markersize=3.0,
                     legend_loc=None,
                     bn_stats_iters=5, bn_stats_batch=128):
    xs = [ i * (1.0 / num_plots) for i in range(0, num_plots+1) ]
    for pair, label, color, marker, linestyle in zip(param_pairs, labels, colors, markers, linestyles):
        ys = []
        w1, w2 = pair
        for x in xs:
            p = {k: (1.0 - x)*w1[k] + x*w2[k] for k in w1}
            set_params_(learner.model, p)

            learner.model.train()
            reset_bn_stats_(learner.model)
            update_bn_stats_(learner, bn_stats_iters, bn_stats_batch)
            learner.model.eval()
            res = learner.evaluate(dataset_type='val')
            ys.append(res['loss'])
            print(f'x = {x}; loss = {res["loss"]}')
        plt.plot(xs, ys,
                label=label, linewidth=linewidth, linestyle=linestyle,
                marker=marker, markersize=markersize,
                color=color)
    plt.xlabel('Linear Interpolation')
    plt.ylabel('Val Loss')
    plt.grid(color='lightgray')
    if legend_loc is not None:
        plt.legend(loc=legend_loc)
    else:
        plt.legend()
    plt.savefig(pdf_path, bbox_inches='tight')

def plot_heatmap(pdf_path, learner,
                w1, label1, w2, label2, w3, label3,
                color1='orange', color2='orange', color3='orange',
                marker1='*', marker2='^', marker3='P',
                x_min=-50, y_min=-50,
                max_loss=None,
                bn_stats_iters=5, bn_stats_batch=128):
    def vec(p):
        return torch.cat([p[k].flatten() for k in p])
    def sub_param(p1, p2):
        return {k: p1[k] - p2[k] for k in p1}
    def mul_param(c, p):
        return {k: c*p[k] for k in p}

    u = sub_param(w2, w1)
    c = (torch.dot(vec(w3)-vec(w1), vec(w2)-vec(w1)) / torch.dot(vec(w2)-vec(w1), vec(w2)-vec(w1))).item()
    v = {k: w3[k] - w1[k] - c*(w2[k] - w1[k]) for k in w3}
    u_abs = vec(u).norm().item()
    v_abs = vec(v).norm().item()
    u = mul_param((1.0 / u_abs), u)
    v = mul_param((1.0 / v_abs), v)

    delta = 1.0
    u_v_abs = max(u_abs, v_abs)
    xs = [i*delta for i in range(x_min, int(u_v_abs * 1.4), 20)]
    ys = [i*delta for i in range(y_min, int(u_v_abs * 1.4), 20)]

    if max_loss is None:
        max_loss = float('inf')
    zs = []
    for y in ys:
        zs.append([])
        for x in xs:
            p = {k: w1[k] + x*u[k] + y*v[k] for k in w1}
            set_params_(learner.model, p)

            learner.model.train()
            reset_bn_stats_(learner.model)
            update_bn_stats_(learner, bn_stats_iters, bn_stats_batch)
            learner.model.eval()
            res = learner.evaluate(dataset_type='val')
            zs[-1].append(min(res['loss'], max_loss))
            print(f'(x, y) = ({x}, {y}), z = {res["loss"]}')

    print(f'|u| = {u_abs}')
    print(f'|v| = {v_abs}')
    print(f'c = {c}')
    print('xs:', xs)
    print('ys:', ys)
    print('zs:', zs)

    x1, y1 = 0, 0
    x2, y2 = u_abs, 0
    x3, y3 = c * u_abs, v_abs
    _plot_heatmap(pdf_path, xs, ys, zs,
                  mark_lis=[(x1, y1, marker1, color1, label1),
                            (x2, y2, marker2, color2, label2),
                            (x3, y3, marker3, color3, label3)])

def _plot_heatmap(filename,
               xs, ys, zs,
               mark_lis=[],  # [(x, y, marker, color, label), ...]
               ):
    #X, Y = np.meshgrid(xs, ys)
    delta = (xs[-1] - xs[0]) / len(xs)

    Z = np.array(zs)

    cmap = plt.get_cmap('viridis', 100)
    extent = [xs[0]-(delta/2), xs[-1]+(delta/2), ys[0]-(delta/2), ys[-1]+(delta/2)]
    plt.imshow(Z, extent=extent, origin='lower', cmap=cmap, norm=LogNorm())
    plt.xlabel('u')
    plt.ylabel('v')
    for (x, y, marker, color, label) in mark_lis:
        plt.plot(x, y, marker=marker, markersize=8, color=color, label=label)

    plt.colorbar()
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')

def section_4_3_cifar10_conv8(cfg, outman, prefix, gpu_id):
    # ======== Plot data =========
    plot_name = 'section_4_3_cifar10_conv8'
    exp_name = 'cifar10_conv8_sgd'
    model_infos = {
        'source': {
            'label': 'Source',
            'marker': '^',
            'color': 'C1',
            'param': None,
        },
        'source_permuted': {
            'label': 'Source*',
            'marker': 'v',
            'color': 'C1',
            'param': None,
        },
        'transferred': {
            'label': 'GMT',
            'marker': '*',
            'color': 'C1',
            'param': None,
        },
        'target_init': {
            'label': 'Target Init',
            'marker': 'o',
            'color': 'C1',
            'param': None,
        },
        'oracle': {
            'label': 'Oracle',
            'marker': 'X',
            'color': 'C1',
            'param': None,
        },
        'noperm': {
            'label': 'Naive',
            'marker': 'o',
            'color': 'C1',
            'param': None,
        },
        'target': {
            'label': 'Trained',
            'marker': '^',
            'color': 'C1',
            'param': None,
        },
        'gmt_finegrained': {
            'label': 'GMT (per epoch)',
            'marker': '*',
            'color': 'C1',
            'param': None,
        },
        'source_permuted_finegrained': {
            'label': 'Source (Permuted)',
            'marker': 'v',
            'color': 'C1',
            'param': None,
        },
        'source_permuted_oracle': {
            'label': 'Source**',
            'marker': 'v',
            'color': 'C1',
            'param': None,
        },
    }
    # ============================

    plot_outman = OutputManager(cfg['output_dir'], plot_name, prefix_hashing=True)
    plot_prefix = prefix
    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    cfgs = cfg['__other_configs__']
    learner = ImageClassification(outman, cfg, device, cfg['data_parallel'], silent=True)

    # ======== Load Params =========
    key = 'source'
    exp_name = 'cifar10_conv8_sgd'
    pref = 'epoch_60--lr_0.05--width_factor_1.0--seed_101--'
    assert key in model_infos
    outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
    ckp = outman.load_checkpoint(prefix=f'dump.{pref}', ext='pth')
    learner.model.load_state_dict(ckp.model_state_dict)
    model_infos[key]['param'] = {k: p.clone().detach() for k, p in learner.model.named_parameters()}

    key = 'target'
    exp_name = 'cifar10_conv8_sgd'
    pref = 'epoch_60--lr_0.05--width_factor_1.0--seed_102--'
    assert key in model_infos
    outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
    ckp = outman.load_checkpoint(prefix=f'dump.{pref}', ext='pth')
    learner.model.load_state_dict(ckp.model_state_dict)
    model_infos[key]['param'] = {k: p.clone().detach() for k, p in learner.model.named_parameters()}

    key = 'target_init'
    exp_name = 'cifar10_conv8_sgd'
    pref = 'epoch_60--lr_0.05--width_factor_1.0--seed_102--'
    assert key in model_infos
    outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
    ckp = outman.load_checkpoint(prefix=f'epoch-1.{pref}', ext='pth') # initial parameter
    learner.model.load_state_dict(ckp.model_state_dict)
    model_infos[key]['param'] = {k: p.clone().detach() for k, p in learner.model.named_parameters()}

    key = 'oracle'
    exp_name = 'transfer_init-cifar10_conv8_usetarget'
    pref = '24e9aa9d3156d3157e23751b37f3d5ab'
    assert key in model_infos
    outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
    tr_ckp = outman.load_checkpoint(prefix=pref, ext='pth', name='params')
    set_params_(learner.model, tr_ckp.best_param)
    model_infos[key]['param'] = {k: p.clone().detach() for k, p in learner.model.named_parameters()}

    key = 'noperm'
    exp_name = 'transfer_init-cifar10_conv8_noperm'
    pref = '24e9aa9d3156d3157e23751b37f3d5ab'
    assert key in model_infos
    outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
    tr_ckp = outman.load_checkpoint(prefix=pref, ext='pth', name='params')
    set_params_(learner.model, tr_ckp.best_param)
    model_infos[key]['param'] = {k: p.clone().detach() for k, p in learner.model.named_parameters()}

    if True:
        key = 'oracle'
        exp_name = 'transfer_init-cifar10_conv8_usetarget'
        pref = '24e9aa9d3156d3157e23751b37f3d5ab'
        assert key in model_infos
        outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
        tr_ckp = outman.load_checkpoint(prefix=pref, ext='pth', name='params')
        set_params_(learner.model, tr_ckp.best_param)
        perm = tr_ckp.perms[1]
        #assert (len(tr_ckp.perms) == 2) and (tr_ckp.perms[0] is None)
        model_infos[key]['param'] = {k: p.clone().detach() for k, p in learner.model.named_parameters()}

        get_permutation_spec = getattr(utils._weight_matching, cfg['model.config_name'] + '_permutation_spec')
        ps = get_permutation_spec(cfg['bn_affine'])

        # Use the above results (perm)
        key = 'source_permuted_oracle'
        exp_name = 'cifar10_conv8_sgd'
        pref = 'epoch_60--lr_0.05--width_factor_1.0--seed_101--'
        assert key in model_infos
        outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
        ckp = outman.load_checkpoint(prefix=f'dump.{pref}', ext='pth')
        learner.model.load_state_dict(ckp.model_state_dict)
        param = {k: p.clone().detach() for k, p in learner.model.named_parameters()}
        param = apply_permutation(ps, perm, param, device=device)
        model_infos[key]['param'] = param

    if True:
        key = 'transferred'
        exp_name = 'transfer_init-cifar10_conv8'
        pref = '586d67a936cc7de100efbc7007f59919'
        assert key in model_infos
        outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
        tr_ckp = outman.load_checkpoint(prefix=pref, ext='pth', name='params')
        #set_params_(learner.model, tr_ckp.best_param)
        set_params_(learner.model, tr_ckp.tr_param)
        perm = tr_ckp.perms[1]
        assert (len(tr_ckp.perms) == 2) and (tr_ckp.perms[0] is None)
        model_infos[key]['param'] = {k: p.clone().detach() for k, p in learner.model.named_parameters()}

        get_permutation_spec = getattr(utils._weight_matching, cfg['model.config_name'] + '_permutation_spec')
        ps = get_permutation_spec(cfg['bn_affine'])

        # Use the above results (perm)
        key = 'source_permuted'
        exp_name = 'cifar10_conv8_sgd'
        pref = 'epoch_60--lr_0.05--width_factor_1.0--seed_101--'
        assert key in model_infos
        outman = OutputManager(cfg['output_dir'], exp_name, prefix_hashing=False)
        ckp = outman.load_checkpoint(prefix=f'dump.{pref}', ext='pth')
        learner.model.load_state_dict(ckp.model_state_dict)
        param = {k: p.clone().detach() for k, p in learner.model.named_parameters()}
        param = apply_permutation(ps, perm, param, device=device)
        model_infos[key]['param'] = param
    # ============================


    # ========= 1D Plot =============

    # Plot 6
    filename = '1d_lmc_6'
    key_pairs = [
            ('source_permuted_oracle', 'oracle'),
            ('source', 'noperm'),
            ('source_permuted', 'transferred'),
            ]
    labels = ['Source* - Oracle', 'Source* - Naive', 'Source* - GMT']
    colors = ['C3', 'C1', 'C0']
    markers = ['', '', '']
    linestyles = ['-.', '-', '-']
    legend_loc = 'upper left'

    param_pairs = [ (model_infos[key1]['param'], model_infos[key2]['param']) for key1, key2 in key_pairs]
    pdf_path = plot_outman.get_abspath(plot_prefix, 'pdf', name=filename)
    plot_1dlandscape(pdf_path, learner, param_pairs, labels, colors, markers, linestyles,
                     legend_loc=legend_loc,
                     bn_stats_iters=cfg['bn_stats_iters'], bn_stats_batch=cfg['bn_stats_batch'])
    plot_outman.print(f'Saved plot at: {pdf_path}', prefix=plot_prefix)
    plt.clf()

    # Plot 5
    filename = '1d_lmc_5'
    key_pairs = [
            ('target', 'oracle'),
            ('target', 'noperm'),
            ('target', 'transferred'),
            ]
    labels = ['Trained - Oracle', 'Trained - Naive', 'Trained - GMT']
    colors = ['C3', 'C1', 'C0']
    markers = ['', '', '']
    linestyles = ['-.', '-', '-']
    legend_loc = 'upper left'

    param_pairs = [ (model_infos[key1]['param'], model_infos[key2]['param']) for key1, key2 in key_pairs]
    pdf_path = plot_outman.get_abspath(plot_prefix, 'pdf', name=filename)
    plot_1dlandscape(pdf_path, learner, param_pairs, labels, colors, markers, linestyles,
                     bn_stats_iters=cfg['bn_stats_iters'], bn_stats_batch=cfg['bn_stats_batch'])
    plot_outman.print(f'Saved plot at: {pdf_path}', prefix=plot_prefix)
    plt.clf()

    # ========= 2D Plot =============
    x_min, y_min = -30, -30
    max_loss = None

    # Plot 4
    filename = 'landscape4'
    key1 = 'target'
    key2 = 'oracle'
    key3 = 'transferred'
    assert (key1 in model_infos) and (key2 in model_infos) and (key3 in model_infos)

    w1, w2, w3 = model_infos[key1]['param'], model_infos[key2]['param'], model_infos[key3]['param']
    assert (w1 is not None) and (w2 is not None) and (w3 is not None)
    label1, label2, label3 = model_infos[key1]['label'], model_infos[key2]['label'], model_infos[key3]['label']
    marker1, marker2, marker3 = model_infos[key1]['marker'], model_infos[key2]['marker'], model_infos[key3]['marker']
    color1, color2, color3 = model_infos[key1]['color'], model_infos[key2]['color'], model_infos[key3]['color']

    pdf_path = plot_outman.get_abspath(plot_prefix, 'pdf', name=filename)
    plot_heatmap(pdf_path, learner,
                 w1, label1,  w2, label2, w3, label3,
                 marker1=marker1, marker2=marker2, marker3=marker3,
                 color1=color1, color2=color2, color3=color3,
                 x_min=x_min, y_min=y_min,
                 max_loss=max_loss,
                 bn_stats_iters=cfg['bn_stats_iters'], bn_stats_batch=cfg['bn_stats_batch'])
    plot_outman.print(f'Saved plot at: {pdf_path}', prefix=plot_prefix)
    plt.clf()

