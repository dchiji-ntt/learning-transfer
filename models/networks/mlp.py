
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.seed import set_random_seed

class MLP(nn.Module):
    def __init__(self, dataset_cfg, model_cfg, cfg):
        super(MLP, self).__init__()

        self.image_size = dataset_cfg['image_size']
        self.num_channels = dataset_cfg['num_channels']
        self.num_classes = dataset_cfg['num_classes']
        self.width = model_cfg['default_width']
        self.width_factor = cfg['width_factor']

        self.linear1 = nn.Linear(self.num_channels * self.image_size ** 2, int(self.width * self.width_factor), bias=False)
        self.linear2 = nn.Linear(int(self.width * self.width_factor), self.num_classes, bias=False)

    def forward(self, x):
        out = x.view(x.size(0), self.num_channels * self.image_size ** 2)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
