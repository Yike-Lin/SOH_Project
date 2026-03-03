from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

from utils.scaler import Scaler



class XJTUDataset:
    """
    初始化,保存一些配置参数
    """
    def __init__(self , args):
        self.root = 'data/XUTU'
        self.max_capacity = 2.0

        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.seed = args.random_seed
        self.batch = args. batch                        # 实验批次
        self.batch_size = args.batch_size               # 训练批次


    """
    归一化
    """
    def _normalize(self , data:np.ndarray) -> np.ndarray:
        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            data_norm = scaler.standard
        else:
            data_norm = scaler.minmax(feature_range=self.minmax_range)
        return data_norm

