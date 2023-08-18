import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor, Resize, Compose, ColorJitter, RandomResizedCrop, RandomHorizontalFlip, Normalize, CenterCrop, Pad
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR

from utils.subset_dataset import SubsetDataset, random_split
from utils.schedulers import CustomCosineLR
import utils.datasets
from models.networks.resnet import ResNet
from models.networks.convs import Conv4, Conv6, Conv8
from models.networks.mlp import MLP
from pytorch_fgvc_dataset.cub2011 import Cub2011

import random
import os
import copy


class ImageClassification(object):
    def __init__(self, outman, cfg, device, data_parallel, silent=False):
        self.outman = outman
        self.cfg = copy.deepcopy(cfg)
        self.device = device
        self.data_parallel = data_parallel
        self.silent = silent

        self.debug_max_iters = self.cfg['debug_max_iters']
        self.train_augmentation = self.cfg['train_augmentation']
        self.dataset_cfg = self.cfg['__other_configs__'][self.cfg['dataset.config_name']]

        self.model_cfg = self.cfg['__other_configs__'][self.cfg['model.config_name']]

        self.train_dataset, self.val_dataset, self.test_dataset = self._get_datasets()
        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.scheduler = self._get_scheduler()

    def train(self, epoch, total_iters, before_callback=None, after_callback=None):
        self.model.train()

        batch_size = self.cfg['batch_size']
        num_workers = self.cfg['num_workers']
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

        results = []
        total_count = 0
        total_loss = 0.
        correct = 0
        if self.debug_max_iters is None:
            iters_per_epoch = len(dataloader)
        else:
            iters_per_epoch = min(len(dataloader), self.debug_max_iters)

        # for the case of self.scheduler == CustomCosineLR
        step_before_train = hasattr(self.scheduler, "step_before_train") and self.scheduler.step_before_train
        if step_before_train:
            try:
                self.scheduler.step(epoch=epoch)
            except:
                self.scheduler.step()

        for _it, (inputs, targets) in enumerate(dataloader):
            if self.debug_max_iters is not None and _it >= self.debug_max_iters:
                break

            if before_callback is not None:
                before_callback(self.model, epoch, total_iters, iters_per_epoch)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            _, predicted = outputs.max(1)
            total_count += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            results.append({
                'mean_loss': loss.item(),
                })

            total_loss += loss.item()
            total_iters += 1

            if after_callback is not None:
                after_callback(self.model, epoch, total_iters, iters_per_epoch)

        if not step_before_train:
            try:
                self.scheduler.step(epoch=epoch)
            except:
                self.scheduler.step()

        self.model.eval()

        return {
                'iterations': total_iters,
                'per_iteration': results,
                'loss': total_loss / total_iters,
                'moving_accuracy': (correct / total_count) * 100.0
                }

    def evaluate(self, dataset_type='val', max_iters=None):
        self.model.eval()

        batch_size = self.cfg['batch_size_eval']
        num_workers = self.cfg['num_workers']
        if dataset_type == 'val':
            dataloader = DataLoader(self.val_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
        elif dataset_type == 'test':
            dataloader = DataLoader(self.test_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
        elif dataset_type == 'train':
            dataloader = DataLoader(self.train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
        else:
            raise NotImplementedError

        if self.debug_max_iters is not None:
            max_iters = self.debug_max_iters

        results = []
        total_count = 0
        total_loss = 0.
        total_iters = 0
        correct = 0
        for _it, (inputs, targets) in enumerate(dataloader):
            if max_iters is not None and _it >= max_iters:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total_count += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_loss += loss.item()
                total_iters += 1
        return {
                'loss': total_loss / total_iters,
                'accuracy': (correct / total_count) * 100.0,
                }

    def _get_datasets(self):
        dataset_dir = self.cfg['dataset_dir']
        max_size = self.cfg['max_train_dataset_size']
        dataset_download = self.cfg['dataset_download']
        dataset_classname = self.dataset_cfg['class']
        data_type = self.dataset_cfg['data_type']

        if dataset_classname in ['CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet', 'StanfordCars']:
            dataset_class = getattr(torchvision.datasets, dataset_classname)
        elif dataset_classname in ['Cub2011']:
            dataset_class = Cub2011
        elif dataset_classname in ['DummyImageDataset']:
            dataset_class = getattr(utils.datasets, dataset_classname)
        else:
            raise NotImplementedError

        if data_type == 'image':
            image_size = self.dataset_cfg['image_size']
            train_val_split = self.cfg['train_val_split']

            transform_train = self._create_transform(image_size, train=True)
            transform_val = self._create_transform(image_size, train=False)

            if dataset_class == torchvision.datasets.ImageNet:
                trainval_dataset = dataset_class(os.path.join(dataset_dir, 'imagenet'),
                                                 split='train',
                                                 transform=None) # to be specified later
            elif dataset_class == torchvision.datasets.StanfordCars:
                trainval_dataset = dataset_class(dataset_dir,
                                                 split='train',
                                                 transform=None, # to be specified later
                                                 download=True,)
            elif dataset_class == Cub2011:
                trainval_dataset = dataset_class(dataset_dir,
                                                 train=True,
                                                 transform=None, # to be specified later
                                                 download=True,)
            elif dataset_class == utils.datasets.DummyImageDataset:
                trainval_dataset = dataset_class(self.dataset_cfg['num_data'],
                                                 self.dataset_cfg['num_channels'],
                                                 self.dataset_cfg['image_size'],
                                                 self.dataset_cfg['num_classes'])
            else:
                trainval_dataset = dataset_class(dataset_dir,
                                                 train=True,
                                                 transform=None, # to be specified later
                                                 download=dataset_download)
            if self.dataset_cfg['specify_classes'] is not None:
                self._filter_dataset_by_specified_classes(trainval_dataset, self.dataset_cfg['specify_classes'])

            size = len(trainval_dataset)
            val_size = int(size * train_val_split)
            train_size = min(size - val_size,
                             max_size if max_size is not None else size)
            gen = torch.Generator()
            gen.manual_seed(777)
            train_subset, val_subset, _ = random_split(trainval_dataset,
                                                [train_size, val_size, size-(train_size+val_size)],
                                                generator=gen)
            if not self.silent:
                self.outman.print('Train/val dataset size:', size)
                self.outman.print('Train dataset size:', len(train_subset),
                            ', Val dataset size:', len(val_subset))

            train_dataset = SubsetDataset(train_subset, transform=transform_train)
            val_dataset = SubsetDataset(val_subset, transform=transform_val)
            if dataset_class == torchvision.datasets.ImageNet:
                test_dataset = dataset_class(os.path.join(dataset_dir, 'imagenet'),
                                             split='val',
                                             transform=transform_val)
            elif dataset_class == torchvision.datasets.StanfordCars:
                test_dataset = dataset_class(dataset_dir,
                                             split='test',
                                             transform=transform_val,
                                             download=True,)
            elif dataset_class == Cub2011:
                test_dataset = dataset_class(dataset_dir,
                                             train=False,
                                             transform=transform_val, # to be specified later
                                             download=True,)
            elif dataset_class == utils.datasets.DummyImageDataset:
                test_dataset = dataset_class(self.dataset_cfg['num_data'],
                                             self.dataset_cfg['num_channels'],
                                             self.dataset_cfg['image_size'],
                                             self.dataset_cfg['num_classes'])
            else:
                test_dataset = dataset_class(dataset_dir,
                                             train=False,
                                             transform=transform_val,
                                             download=dataset_download)
            if self.dataset_cfg['specify_classes'] is not None:
                self._filter_dataset_by_specified_classes(test_dataset, self.dataset_cfg['specify_classes'])
        else:
            raise NotImplementedError

        return train_dataset, val_dataset, test_dataset

    def _filter_dataset_by_specified_classes(self, dataset, class_ids):
        filtered_idxs = []
        target_map = {id: k for k, id in enumerate(class_ids)}

        for idx, target in enumerate(dataset.targets):
            if target in class_ids:
                filtered_idxs.append(idx)
        dataset.data = dataset.data[filtered_idxs, :]
        dataset.targets = [target_map[t] for i, t in enumerate(dataset.targets) if i in filtered_idxs]
        dataset.classes = [c for i, c in enumerate(dataset.classes) if i in class_ids]
        dataset.class_to_idx = {c: i for i, c in enumerate(dataset.classes)}

    def _create_transform(self, image_size, train=False):
        dataset_class = self.dataset_cfg['class']
        norm_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        jitter_param = dict(brightness=0.4, contrast=0.4, saturation=0.4)

        if train and self.train_augmentation:
            if dataset_class in ['MNIST']:
                train_transform = Compose([
                                Resize((image_size, image_size)),
                                ToTensor(),
                                Normalize((0.1307,), (0.3081,)),
                               ])
            elif dataset_class in ['DummyImageDataset']:
                train_transform = None
            else:
                train_transform = Compose([
                                RandomResizedCrop((image_size, image_size)),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                Normalize(**norm_param),
                               ])
            if self.cfg['padding_before_crop']:  # this should be used in CIFAR-10 training
                train_transform.transforms.insert(0, Pad(4))
            return train_transform
        else:
            if dataset_class in ['ImageNet', 'StanfordCars', 'Cub2011']:
                return Compose([Resize(256),
                                CenterCrop(image_size),
                                ToTensor(),
                                Normalize(**norm_param)
                               ])
            elif dataset_class in ['CIFAR10', 'CIFAR100']:
                return Compose([Resize((image_size, image_size)),
                                ToTensor(),
                                Normalize(**norm_param)
                               ])
            elif dataset_class in ['MNIST']:
                return Compose([Resize((image_size, image_size)),
                                ToTensor(),
                                Normalize((0.1307,), (0.3081,)),
                               ])
            elif dataset_class in ['DummyImageDataset']:
                return None
            else:
                raise NotImplementedError

    def _get_model(self, model_cfg=None):
        if model_cfg is None:
            model_cfg = self.model_cfg

        if model_cfg['class'] == 'ResNet':
            model = ResNet(self.dataset_cfg, model_cfg, self.cfg)
        elif model_cfg['class'] == 'MLP':
            model = MLP(self.dataset_cfg, model_cfg, self.cfg)
        elif model_cfg['class'] == 'Conv4':
            model = Conv4(self.dataset_cfg, model_cfg, self.cfg)
        elif model_cfg['class'] == 'Conv6':
            model = Conv6(self.dataset_cfg, model_cfg, self.cfg)
        elif model_cfg['class'] == 'Conv8':
            model = Conv8(self.dataset_cfg, model_cfg, self.cfg)
        else:
            raise NotImplementedError

        if self.data_parallel:
            gpu_ids = list(range(self.cfg['num_gpus']))
            return DataParallel(model, gpu_ids)
        else:
            return model

    def _get_optimizer(self):
        optim_name = self.cfg['optimizer']
        lr = self.cfg['lr']
        weight_decay = self.cfg['weight_decay']
        params = [p for p in self.model.parameters()]
        return self._new_optimizer(optim_name, params, lr, weight_decay)

    def _get_criterion(self):
        return nn.CrossEntropyLoss()

    def _new_optimizer(self, name, params, lr, weight_decay, momentum=0.9):
        if name == 'AdamW':
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == 'SGD':
            return torch.optim.SGD(params, lr=lr,
                                   momentum=self.cfg['sgd_momentum'], weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def _get_scheduler(self):
        class null_scheduler(object):
            def __init__(self, *args, **kwargs):
                return
            def step(self, *args, **kwargs):
                return
            def state_dict(self):
                return {}
            def load_state_dict(self, dic):
                return

        if self.cfg['lr_scheduler'] is None:
            return null_scheduler()
        elif self.cfg['lr_scheduler'] == 'CustomCosineLR':
            total_epoch = self.cfg['epoch']
            init_lr = self.cfg['lr']
            warmup_epochs = self.cfg['warmup_epochs']
            ft_epochs = self.cfg['finetuning_epochs']
            ft_lr = self.cfg['finetuning_lr']
            return CustomCosineLR(self.optimizer, init_lr, total_epoch, warmup_epochs, ft_epochs, ft_lr)
        elif self.cfg['lr_scheduler'] == 'MultiStepLR':
            return MultiStepLR(self.optimizer, milestones=self.cfg['lr_milestones'], gamma=self.cfg['multisteplr_gamma'])
        else:
            raise NotImplementedError

