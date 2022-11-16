# Select you GPU
I_GPU = 0
# Uncomment to use autoreload
# %load_ext autoreload
# %autoreload 2

import os
import os.path as osp
import sys
import torch
import numpy as np
from time import time
from omegaconf import OmegaConf
start = time()
import warnings
warnings.filterwarnings('ignore')

torch.cuda.set_device(I_GPU)
DIR = os.path.dirname(os.getcwd())
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, DIR)

from torch_points3d.utils.config import hydra_read
from torch_geometric.data import Data
from torch_points3d.core.multimodal.data import MMData, MMBatch
from torch_points3d.visualization.multimodal_data import visualize_mm_data
from torch_points3d.core.multimodal.image import SameSettingImageData, ImageData
from torch_points3d.datasets.segmentation.multimodal.scannet import ScannetDatasetMM
from torch_points3d.datasets.segmentation.scannet import CLASS_COLORS, CLASS_NAMES

import pykeops

if __name__ == '__main__':
    # Uncomment to clean previous pykeops builds with CUDA
    pykeops.clean_pykeops()
    
    # Set your dataset root directory, where the data was/will be downloaded
    DATA_ROOT = '/project/fsun/dvata'

    dataset_config = 'segmentation/multimodal/Feng/scannet-neucon-smallres-m2f.yaml'   
    models_config = 'segmentation/multimodal/sparseconv3d'    # model family
    model_name = 'Res16UNet34-L4-early'                       # specific model

    overrides = [
        'task=segmentation',
        f'data={dataset_config}',
        f'models={models_config}',
        f'model_name={model_name}',
        f'data.dataroot={DATA_ROOT}',
    ]

    cfg = hydra_read(overrides)
    # print(OmegaConf.to_yaml(cfg))


    # Dataset instantiation
    start = time()
    dataset = ScannetDatasetMM(cfg.data)
    print(dataset)
    print(f"Time = {time() - start:0.1f} sec.")
