import os
from pathlib import Path
from typing import Any, Dict, List, Iterator, Optional, Tuple
import torch
from torchvision.datasets.folder import ImageFolder
import csv
import subprocess

import torchvision.datasets

class DummyImageDataset():
    def __init__(self, num_data, channel, image_size, label_size, **kwargs):
        self.num_data = num_data
        self.channel = channel
        self.image_size = image_size
        self.label_size = label_size

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        image = torch.rand(self.channel, self.image_size, self.image_size)
        label = torch.zeros(self.label_size)
        label_idx = int(image.sum().item() * 999) % self.label_size
        return image, label_idx

