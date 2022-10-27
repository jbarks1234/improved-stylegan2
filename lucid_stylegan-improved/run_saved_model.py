import os
import sys
import math
# import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter3d

import torchvision
from torchvision import transforms
from version import __version__
from diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize

import random
from retry.api import retry_call
from datetime import datetime

import torch.multiprocessing as mp
import torch.distributed as dist

from PIL import Image
from pathlib import Path

config_path = "./models/default/config.json"

def config(self):
    return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp,
            'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size,
            'attn_layers': self.attn_layers, 'no_const': self.no_const}

def load_config(self):
    config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
    self.image_size = config['image_size']
    self.network_capacity = config['network_capacity']
    self.transparent = config['transparent']
    self.fq_layers = config['fq_layers']
    self.fq_dict_size = config['fq_dict_size']
    self.fmap_max = config.pop('fmap_max', 512)
    self.attn_layers = config.pop('attn_layers', [])
    self.no_const = config.pop('no_const', False)
    self.lr_mlp = config.pop('lr_mlp', 0.1)
    del self.GAN
    self.init_GAN()