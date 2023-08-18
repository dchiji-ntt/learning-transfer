
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from utils.seed import set_random_seed

class Conv4(nn.Module):
    def __init__(self, dataset_cfg, model_cfg, cfg):
        super(Conv4, self).__init__()
        
        assert dataset_cfg['image_size'] == 32
        assert dataset_cfg['num_channels'] == 3

        self.num_classes = dataset_cfg['num_classes']
        self.factor = cfg['width_factor']

        self.convs = nn.Sequential(
            nn.Conv2d(3, int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(64 * self.factor), int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(int(64 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(128 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.linear = nn.Sequential(
            nn.Conv2d(int(128 * self.factor) * 8 * 8, int(256 * self.factor), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), self.num_classes, kernel_size=1)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), int(128 * self.factor) * 8 * 8, 1, 1)
        out = self.linear(out)
        return out.squeeze()

class Conv6(nn.Module):
    def __init__(self, dataset_cfg, model_cfg, cfg):
        super(Conv6, self).__init__()
        
        assert dataset_cfg['image_size'] == 32
        assert dataset_cfg['num_channels'] == 3

        self.num_classes = dataset_cfg['num_classes']
        self.factor = cfg['width_factor']

        self.convs = nn.Sequential(
            nn.Conv2d(3, int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(64 * self.factor), int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(int(64 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(128 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(int(128 * self.factor), int(256 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d((2, 2))
            nn.MaxPool2d((8, 8))
        )

        self.linear = nn.Sequential(
            #nn.Conv2d(int(256 * self.factor) * 4 * 4, int(256 * self.factor), kernel_size=1),
            nn.Conv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), self.num_classes, kernel_size=1)
        )

    def forward(self, x):
        out = self.convs(x)
        #out = out.view(out.size(0), int(256 * self.factor) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()

    def initialize_head(self, seed):
        set_random_seed(seed)
        self.linear[-1].reset_parameters()


class Conv8(nn.Module):
    def __init__(self, dataset_cfg, model_cfg, cfg):
        super(Conv8, self).__init__()
        
        assert dataset_cfg['image_size'] == 32
        assert dataset_cfg['num_channels'] == 3

        self.num_classes = dataset_cfg['num_classes']
        self.factor = cfg['width_factor']

        self.convs = nn.Sequential(
            nn.Conv2d(3, int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(64 * self.factor), int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(int(64 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(128 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(int(128 * self.factor), int(256 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(int(256 * self.factor), int(512 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(512 * self.factor), int(512 * self.factor), kernel_size=3,
                         stride=1, padding=1),
            nn.ReLU(),
            # TODO: original is nn.MaxPool2d((2, 2)), but we have to modify it due to permutation issue
            nn.MaxPool2d((4, 4))
        )

        self.linear = nn.Sequential(
            nn.Conv2d(int(512 * self.factor), int(256 * self.factor), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * self.factor), self.num_classes, kernel_size=1)
        )

    def forward(self, x):
        out = self.convs(x)
        #out = out.view(out.size(0), int(256 * self.factor) * 4 * 4, 1, 1)
        out = out.view(out.size(0), int(512 * self.factor), 1, 1)
        out = self.linear(out)
        return out.squeeze()

    def initialize_head(self, seed):
        set_random_seed(seed)
        self.linear[-1].reset_parameters()

    def load_state_dict_without_head(self, state_dict):
        state_dict_copy = copy.deepcopy(state_dict)
        state_dict_copy['linear.4.weight'] = self.state_dict()['linear.4.weight']
        state_dict_copy['linear.4.bias'] = self.state_dict()['linear.4.bias']
        self.load_state_dict(state_dict_copy)


