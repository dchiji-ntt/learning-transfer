# The code is from https://github.com/samuela/git-re-basin (MIT License)

from collections import defaultdict
from typing import NamedTuple

import torch
import numpy as np
#import jax.numpy as jnp
#from jax import random
import random
from scipy.optimize import linear_sum_assignment
from numpy import linalg as LA

rngmix = lambda rng, x: random.fold_in(rng, hash(x))

def set_params_(model, param_dic):
    for name, p in model.named_parameters():
        device = p.data.device
        if isinstance(param_dic[name], np.ndarray):
            p.data = torch.from_numpy(param_dic[name]).to(device)
        else:
            p.data = param_dic[name].clone().detach().to(device)

class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def conv4_permutation_spec(bn_affine):
    return permutation_spec_from_axes_to_perm({
        'convs.0.weight': ('c64-1', None, None, None),
        'convs.0.bias': ('c64-1', ),
        'convs.2.weight': ('c64-2', 'c64-1', None, None),
        'convs.2.bias': ('c64-2', ),
        'convs.5.weight': ('c128-1', 'c64-2', None, None),
        'convs.5.bias': ('c128-1', ),
        'convs.7.weight': (None, 'c128-1', None, None),
        'convs.7.bias': (None, ),
        'linear.0.weight': ('c256-1', None, None, None),
        'linear.0.bias': ('c256-1', ),
        'linear.2.weight': ('c256-2', 'c256-1', None, None),
        'linear.2.bias': ('c256-2', ),
        'linear.4.weight': (None, 'c256-2', None, None),
        'linear.4.bias': (None, ),
    })

def mlp_permutation_spec(bn_affine):
    return permutation_spec_from_axes_to_perm({
        'linear1.weight': ('c1', None),
        #'linear1.bias': ('c1', ),
        'linear2.weight': (None, 'c1'),
        #'linear2.bias': (None, ),
        })

def conv6_permutation_spec(bn_affine):
    return permutation_spec_from_axes_to_perm({
        'convs.0.weight': ('c64-1', None, None, None),
        'convs.0.bias': ('c64-1', ),
        'convs.2.weight': ('c64-2', 'c64-1', None, None),
        'convs.2.bias': ('c64-2', ),
        'convs.5.weight': ('c128-1', 'c64-2', None, None),
        'convs.5.bias': ('c128-1', ),
        'convs.7.weight': ('c128-2', 'c128-1', None, None),
        'convs.7.bias': ('c128-2', ),
        'convs.10.weight': ('c256-1', 'c128-2', None, None),
        'convs.10.bias': ('c256-1', ),
        'convs.12.weight': ('c256-2', 'c256-1', None, None),
        'convs.12.bias': ('c256-2', ),
        'linear.0.weight': ('c256-3', 'c256-2', None, None),
        'linear.0.bias': ('c256-3', ),
        'linear.2.weight': ('c256-4', 'c256-3', None, None),
        'linear.2.bias': ('c256-4', ),
        'linear.4.weight': (None, 'c256-4', None, None),
        'linear.4.bias': (None, ),
    })

def conv8_permutation_spec(bn_affine):
    return permutation_spec_from_axes_to_perm({
        "convs.0.weight": ('c64-1', None, None, None),
        "convs.0.bias": ('c64-1', ),
        "convs.2.weight": ('c64-2', 'c64-1', None, None),
        "convs.2.bias": ('c64-2', ),
        "convs.5.weight": ('c128-1', 'c64-2', None, None),
        "convs.5.bias": ('c128-1', ),
        "convs.7.weight": ('c128-2', 'c128-1', None, None),
        "convs.7.bias": ('c128-2', ),
        "convs.10.weight": ('c256-1', 'c128-2', None, None),
        "convs.10.bias": ('c256-1', ),
        "convs.12.weight": ('c256-2', 'c256-1', None, None),
        "convs.12.bias": ('c256-2', ),
        "convs.15.weight": ('c512-1', 'c256-2', None, None),
        "convs.15.bias": ('c512-1', ),
        "convs.17.weight": ('c512-2', 'c512-1', None, None),
        "convs.17.bias": ('c512-2', ),
        "linear.0.weight": ('c256-3', 'c512-2', None, None),
        "linear.0.bias": ('c256-3', ),
        "linear.2.weight": ('c256-4', 'c256-3', None, None),
        "linear.2.bias": ('c256-4', ),
        "linear.4.weight": (None, 'c256-4', None, None),
        "linear.4.bias": (None, ),
    })

def resnet18_permutation_spec(bn_affine):
    axes_to_perm = {
        "convnb1.conv.weight": ('c64', None, None, None),
        "convnb1.norm.weight": ('c64', ) if bn_affine else None,
        "convnb1.norm.bias": ('c64', ) if bn_affine else None,
  
        "layer1.0.convnb1.conv.weight": ('c64-1', 'c64', None, None),
        "layer1.0.convnb1.norm.weight": ('c64-1', ) if bn_affine else None,
        "layer1.0.convnb1.norm.bias": ('c64-1', ) if bn_affine else None,
        "layer1.0.convnb2.conv.weight": ('c64', 'c64-1', None, None),
        "layer1.0.convnb2.norm.weight": ('c64', ) if bn_affine else None,
        "layer1.0.convnb2.norm.bias": ('c64', ) if bn_affine else None,

        "layer1.1.convnb1.conv.weight": ('c64-2', 'c64', None, None),
        "layer1.1.convnb1.norm.weight": ('c64-2', ) if bn_affine else None,
        "layer1.1.convnb1.norm.bias": ('c64-2', ) if bn_affine else None,
        "layer1.1.convnb2.conv.weight": ('c64', 'c64-2', None, None),
        "layer1.1.convnb2.norm.weight": ('c64', ) if bn_affine else None,
        "layer1.1.convnb2.norm.bias": ('c64', ) if bn_affine else None,

        "layer2.0.convnb1.conv.weight": ('c128-1', 'c64', None, None),
        "layer2.0.convnb1.norm.weight": ('c128-1', ) if bn_affine else None,
        "layer2.0.convnb1.norm.bias": ('c128-1', ) if bn_affine else None,
        "layer2.0.convnb2.conv.weight": ('c128', 'c128-1', None, None),
        "layer2.0.convnb2.norm.weight": ('c128', ) if bn_affine else None,
        "layer2.0.convnb2.norm.bias": ('c128', ) if bn_affine else None,

        "layer2.0.shortcut.conv.weight": ('c128', 'c64', None, None),
        "layer2.0.shortcut.norm.weight": ('c128', ) if bn_affine else None,
        "layer2.0.shortcut.norm.bias": ('c128', ) if bn_affine else None,

        "layer2.1.convnb1.conv.weight": ('c128-2', 'c128', None, None),
        "layer2.1.convnb1.norm.weight": ('c128-2', ) if bn_affine else None,
        "layer2.1.convnb1.norm.bias": ('c128-2', ) if bn_affine else None,
        "layer2.1.convnb2.conv.weight": ('c128', 'c128-2', None, None),
        "layer2.1.convnb2.norm.weight": ('c128', ) if bn_affine else None,
        "layer2.1.convnb2.norm.bias": ('c128', ) if bn_affine else None,

        "layer3.0.convnb1.conv.weight": ('c256-1', 'c128', None, None),
        "layer3.0.convnb1.norm.weight": ('c256-1', ) if bn_affine else None,
        "layer3.0.convnb1.norm.bias": ('c256-1', ) if bn_affine else None,
        "layer3.0.convnb2.conv.weight": ('c256', 'c256-1', None, None),
        "layer3.0.convnb2.norm.weight": ('c256', ) if bn_affine else None,
        "layer3.0.convnb2.norm.bias": ('c256', ) if bn_affine else None,

        "layer3.0.shortcut.conv.weight": ('c256', 'c128', None, None),
        "layer3.0.shortcut.norm.weight": ('c256', ) if bn_affine else None,
        "layer3.0.shortcut.norm.bias": ('c256', ) if bn_affine else None,

        "layer3.1.convnb1.conv.weight": ('c256-2', 'c256', None, None),
        "layer3.1.convnb1.norm.weight": ('c256-2', ) if bn_affine else None,
        "layer3.1.convnb1.norm.bias": ('c256-2', ) if bn_affine else None,
        "layer3.1.convnb2.conv.weight": ('c256', 'c256-2', None, None),
        "layer3.1.convnb2.norm.weight": ('c256', ) if bn_affine else None,
        "layer3.1.convnb2.norm.bias": ('c256', ) if bn_affine else None,

        "layer4.0.convnb1.conv.weight": ('c512-1', 'c256', None, None),
        "layer4.0.convnb1.norm.weight": ('c512-1', ) if bn_affine else None,
        "layer4.0.convnb1.norm.bias": ('c512-1', ) if bn_affine else None,
        "layer4.0.convnb2.conv.weight": ('c512', 'c512-1', None, None),
        "layer4.0.convnb2.norm.weight": ('c512', ) if bn_affine else None,
        "layer4.0.convnb2.norm.bias": ('c512', ) if bn_affine else None,

        "layer4.0.shortcut.conv.weight": ('c512', 'c256', None, None),
        "layer4.0.shortcut.norm.weight": ('c512', ) if bn_affine else None,
        "layer4.0.shortcut.norm.bias": ('c512', ) if bn_affine else None,

        "layer4.1.convnb1.conv.weight": ('c512-2', 'c512', None, None),
        "layer4.1.convnb1.norm.weight": ('c512-2', ) if bn_affine else None,
        "layer4.1.convnb1.norm.bias": ('c512-2', ) if bn_affine else None,
        "layer4.1.convnb2.conv.weight": ('c512', 'c512-2', None, None),
        "layer4.1.convnb2.norm.weight": ('c512', ) if bn_affine else None,
        "layer4.1.convnb2.norm.bias": ('c512', ) if bn_affine else None,

        "linear.weight": (None, 'c512'),
        "linear.bias": (None, ),
    }
    for key in list(axes_to_perm.keys()):
        if axes_to_perm[key] is None:
            del axes_to_perm[key]
    return permutation_spec_from_axes_to_perm(axes_to_perm)

def vgg16_permutation_spec() -> PermutationSpec:
  return permutation_spec_from_axes_to_perm({
      "Conv_0/kernel": (None, None, None, "P_Conv_0"),
      **{f"Conv_{i}/kernel": (None, None, f"P_Conv_{i-1}", f"P_Conv_{i}")
         for i in range(1, 13)},
      **{f"Conv_{i}/bias": (f"P_Conv_{i}", )
         for i in range(13)},
      **{f"LayerNorm_{i}/scale": (f"P_Conv_{i}", )
         for i in range(13)},
      **{f"LayerNorm_{i}/bias": (f"P_Conv_{i}", )
         for i in range(13)},
      "Dense_0/kernel": ("P_Conv_12", "P_Dense_0"),
      "Dense_0/bias": ("P_Dense_0", ),
      "Dense_1/kernel": ("P_Dense_0", "P_Dense_1"),
      "Dense_1/bias": ("P_Dense_1", ),
      "Dense_2/kernel": ("P_Dense_1", None),
      "Dense_2/bias": (None, ),
  })

def resnet20_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
  norm = lambda name, p: {f"{name}/scale": (p, ), f"{name}/bias": (p, )}
  dense = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out, )}

  # This is for easy blocks that use a residual connection, without any change in the number of channels.
  easyblock = lambda name, p: {
      **conv(f"{name}/conv1", p, f"P_{name}_inner"),
      **norm(f"{name}/norm1", f"P_{name}_inner"),
      **conv(f"{name}/conv2", f"P_{name}_inner", p),
      **norm(f"{name}/norm2", p)
  }

  # This is for blocks that use a residual connection, but change the number of channels via a Conv.
  shortcutblock = lambda name, p_in, p_out: {
      **conv(f"{name}/conv1", p_in, f"P_{name}_inner"),
      **norm(f"{name}/norm1", f"P_{name}_inner"),
      **conv(f"{name}/conv2", f"P_{name}_inner", p_out),
      **norm(f"{name}/norm2", p_out),
      **conv(f"{name}/shortcut/layers_0", p_in, p_out),
      **norm(f"{name}/shortcut/layers_1", p_out),
  }

  return permutation_spec_from_axes_to_perm({
      **conv("conv1", None, "P_bg0"),
      **norm("norm1", "P_bg0"),
      #
      **easyblock("blockgroups_0/blocks_0", "P_bg0"),
      **easyblock("blockgroups_0/blocks_1", "P_bg0"),
      **easyblock("blockgroups_0/blocks_2", "P_bg0"),
      #
      **shortcutblock("blockgroups_1/blocks_0", "P_bg0", "P_bg1"),
      **easyblock("blockgroups_1/blocks_1", "P_bg1"),
      **easyblock("blockgroups_1/blocks_2", "P_bg1"),
      #
      **shortcutblock("blockgroups_2/blocks_0", "P_bg1", "P_bg2"),
      **easyblock("blockgroups_2/blocks_1", "P_bg2"),
      **easyblock("blockgroups_2/blocks_2", "P_bg2"),
      #
      **dense("dense", "P_bg2", None),
  })

def resnet50_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
  norm = lambda name, p: {f"{name}/scale": (p, ), f"{name}/bias": (p, )}
  dense = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out, )}

  # This is for easy blocks that use a residual connection, without any change in the number of channels.
  easyblock = lambda name, p: {
      **conv(f"{name}/conv1", p, f"P_{name}_inner1"),
      **norm(f"{name}/norm1", f"P_{name}_inner1"),
      #
      **conv(f"{name}/conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}/norm2", f"P_{name}_inner2"),
      #
      **conv(f"{name}/conv3", f"P_{name}_inner2", p),
      **norm(f"{name}/norm3", p),
  }

  # This is for blocks that use a residual connection, but change the number of channels via a Conv.
  shortcutblock = lambda name, p_in, p_out: {
      **conv(f"{name}/conv1", p_in, f"P_{name}_inner1"),
      **norm(f"{name}/norm1", f"P_{name}_inner1"),
      #
      **conv(f"{name}/conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}/norm2", f"P_{name}_inner2"),
      #
      **conv(f"{name}/conv2", f"P_{name}_inner2", p_out),
      **norm(f"{name}/norm2", p_out),
      #
      **conv(f"{name}/shortcut/layers_0", p_in, p_out),
      **norm(f"{name}/shortcut/layers_1", p_out),
  }

  return permutation_spec_from_axes_to_perm({
      **conv("conv1", None, "P_bg0"),
      **norm("norm1", "P_bg0"),
      #
      **shortcutblock("blockgroups_0/blocks_0", "P_bg0", "P_bg1"),
      **easyblock("blockgroups_0/blocks_1", "P_bg1"),
      **easyblock("blockgroups_0/blocks_2", "P_bg1"),
      #
      **shortcutblock("blockgroups_1/blocks_0", "P_bg1", "P_bg2"),
      **easyblock("blockgroups_1/blocks_1", "P_bg2"),
      **easyblock("blockgroups_1/blocks_2", "P_bg2"),
      **easyblock("blockgroups_1/blocks_3", "P_bg2"),
      #
      **shortcutblock("blockgroups_2/blocks_0", "P_bg2", "P_bg3"),
      **easyblock("blockgroups_2/blocks_1", "P_bg3"),
      **easyblock("blockgroups_2/blocks_2", "P_bg3"),
      **easyblock("blockgroups_2/blocks_3", "P_bg3"),
      **easyblock("blockgroups_2/blocks_4", "P_bg3"),
      **easyblock("blockgroups_2/blocks_5", "P_bg3"),
      #
      **shortcutblock("blockgroups_3/blocks_0", "P_bg3", "P_bg4"),
      **easyblock("blockgroups_3/blocks_1", "P_bg4"),
      **easyblock("blockgroups_3/blocks_2", "P_bg4"),
      #
      **dense("dense", "P_bg4", None),
  })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
      perm_rep = perm[p].repeat(w.size()[:axis] + w.size()[axis+1:] + (1,))
      perm_rep = torch.unsqueeze(perm_rep, axis)
      perm_rep = torch.transpose(perm_rep, -1, axis)
      perm_rep = torch.squeeze(perm_rep, -1)
      w = torch.gather(w, axis, perm_rep)

  return w

def apply_permutation(ps: PermutationSpec, perm, params, device=None):
  """Apply a `perm` to `params`."""
  if perm is None:
      return {k: params[k].clone().detach() for k in params}
  if device is None:
      device = torch.device('cpu')
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(ps: PermutationSpec,
                    params_a_1,
                    params_b_1,
                    params_a_2=None,
                    params_b_2=None,
                    coeff=1.0,
                    max_iter=100,
                    init_perm=None,
                    normalize=False,
                    silent=False):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a_1[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: np.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  perm_names = list(perm.keys())

  for iteration in range(max_iter):
    progress = False
    #for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
    p_ixs = list(range(len(perm_names)))
    random.shuffle(p_ixs)
    for p_ix in p_ixs:
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = torch.zeros((n, n))
      for wk, axis in ps.perm_to_axes[p]:
        w_a = params_a_1[wk].to('cpu').detach().numpy()
        w_b = get_permuted_param(ps, perm, wk, params_b_1, except_axis=axis)
        w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
        if normalize and (LA.norm(w_a) * LA.norm(w_b)) > 1e-12:
          A += w_a @ w_b.T / (LA.norm(w_a) * LA.norm(w_b))
        else:
          A += w_a @ w_b.T

        if params_a_2 is not None:
          w_a = params_a_2[wk].to('cpu').detach().numpy()
          w_b = get_permuted_param(ps, perm, wk, params_b_2, except_axis=axis)
          w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
          w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
          if normalize and (LA.norm(w_a) * LA.norm(w_b)) > 1e-12:
            A += coeff * (w_a @ w_b.T) / (LA.norm(w_a) * LA.norm(w_b))
          else:
            A += coeff * (w_a @ w_b.T)

      ri, ci = linear_sum_assignment(A, maximize=True)
      assert (ri == np.arange(len(ri))).all()

      oldL = np.vdot(A, np.eye(n)[perm[p]])
      newL = np.vdot(A, np.eye(n)[ci, :])
      if not silent: print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = np.array(ci)

    if not progress:
      break

  return perm

def test_weight_matching():
  """If we just have a single hidden layer then it should converge after just one step."""
  ps = mlp_permutation_spec(num_hidden_layers=1)
  rng = random.PRNGKey(123)
  num_hidden = 10
  shapes = {
      "Dense_0/kernel": (2, num_hidden),
      "Dense_0/bias": (num_hidden, ),
      "Dense_1/kernel": (num_hidden, 3),
      "Dense_1/bias": (3, )
  }
  params_a = {k: random.normal(rngmix(rng, f"a-{k}"), shape) for k, shape in shapes.items()}
  params_b = {k: random.normal(rngmix(rng, f"b-{k}"), shape) for k, shape in shapes.items()}
  perm = weight_matching(rng, ps, params_a, params_b)
  print(perm)

if __name__ == "__main__":
  test_weight_matching()
