
from collections import defaultdict
from typing import NamedTuple

import torch
import numpy as np
#import jax.numpy as jnp
#from jax import random
import random
from scipy.optimize import linear_sum_assignment
from numpy import linalg as LA

from utils._weight_matching import PermutationSpec, get_permuted_param, apply_permutation

def copy_named_parameters(params):
    return {k: p.clone().detach() for k, p in params.items()}

class WeightMatching(object):
    def __init__(self, ps: PermutationSpec, epsilon=1e-7, device=None):
        assert epsilon is not None
        self.ps = ps
        self.epsilon = epsilon
        if device is not None:
            self.device = device
        else:
            self.device = 'cpu'

        self.target_list = []
        self.source_list = []
        self.coeff_list = []

    def add(self, target, source, coeff=1.0):
        """Find a permutation for `source` params to make them match `target` params."""
        self._validate(target)
        self._validate(source)

        self.target_list.append(copy_named_parameters(target))
        self.source_list.append(copy_named_parameters(source))
        self.coeff_list.append(coeff)

    def _validate(self, params):
        for key in params:
            if key not in self.ps.axes_to_perm:
                raise Exception('Permutation spec must contain:', key)
        for key in self.ps.axes_to_perm:
            if key not in params:
                raise Exception('The parameter in permutation spec is missing:', key)

    def calc_loss(self, perm, device):
        loss = 0.0
        for (s, t, c) in zip(self.source_list, self.target_list, self.coeff_list):
            tp = apply_permutation(self.ps, perm, t, device)
            norms = [torch.norm(s[key] - tp[key]) for key in s]
            loss += c * sum(norms)
        return loss

    def solve(self,
              max_iter=100,
              init_perm=None,
              normalize=False,
              silent=False):
        
        perm_sizes = {p: self.target_list[0][axes[0][0]].shape[axes[0][1]] for p, axes in self.ps.perm_to_axes.items()}

        perm = {p: torch.LongTensor(np.arange(n)).to(self.device)
                 for p, n in perm_sizes.items()} if init_perm is None else init_perm
        perm_names = list(perm.keys())

        for iteration in range(max_iter):
            progress = False
            p_ixs = list(range(len(perm_names)))
            random.shuffle(p_ixs)
            for p_ix in p_ixs:
                p = perm_names[p_ix]
                n = perm_sizes[p]
                A = torch.zeros((n, n)).to(self.device)

                for target, source, coeff in  zip(self.target_list, self.source_list, self.coeff_list):
                    for wk, axis in self.ps.perm_to_axes[p]:
                        w_t = target[wk].detach()
                        w_s = get_permuted_param(self.ps, perm, wk, source, except_axis=axis)
                        w_t = torch.moveaxis(w_t, axis, 0).reshape((n, -1))
                        w_s = torch.moveaxis(w_s, axis, 0).reshape((n, -1))
                        if normalize and (torch.norm(w_t) * torch.norm(w_s)) > 1e-12:
                            A += coeff * torch.matmul(w_t, w_s.transpose(0, 1)) / (torch.norm(w_t) * torch.norm(w_s))
                        else:
                            A += coeff * torch.matmul(w_t, w_s.transpose(0, 1))

                ri, ci = linear_sum_assignment(A.numpy(force=True), maximize=True)
                assert (ri == np.arange(len(ri))).all()

                I_n = torch.eye(n).to(self.device)
                oldL = torch.dot(A.flatten(), I_n[perm[p]].flatten())
                newL = torch.dot(A.flatten(), I_n[ci, :].flatten())
                if not silent: print(f"{iteration}/{p}: {newL - oldL}")
                progress = progress or newL > oldL + self.epsilon

                perm[p] = torch.LongTensor(np.array(ci)).to(self.device)

            if not progress:
                break

        return perm
  
