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
    
    """
    解析单块电池的.mat数据
    """
    def _parser_mat_data(self , battery_i_mat):
        # 1. 初始化列表
        data_list = []          # 装时间序列特征    
        cap_list = []           # 装每一圈cycle测出来的实际容量

        # 2. 核心循环：一圈圈拆数据
        for i in range(battery_i_mat.shape[1]): # battery_i_mat.shape[1] 是该电池总循环次数
            cycle_i = battery_i_mat[0 , i]      # cycle_i 是当前第 i 圈的所有数据包

            time = cycle_i['relative_time_min']
            current = cycle_i['current_A']
            voltage = cycle_i['voltage_V']
            temp = cycle_i['temperature_C']

            capacity = cycle_i['capacity'][0]
            cap_list.append(capacity)

            cycle_arr = np.concatenate([time , current , voltage , temp] , axis=0)
            data_list.append(cycle_arr)

        # 3. 所有圈数合并：(4 , L) -> (N , 4 , L)
        data = np.asarray(data_list , dtype=np.float32)
        label = np.asarray(cap_list , dtype=np.float32)

        # 4. 归一化，贴标签
        data = self._normalize(data)
        soh = label / self.max_capacity

        return data,soh
    

    """
    numpy -> torch + 封装 DataLoader
    """
    def _encapsulation(self ,train_x , train_y , test_x , test_y):

        # 1. Numpy -> Pytorch Tensor
        train_x = torch.from_numpy(train_x)                     # 形状仍然是 (N, 4, L) 或 (N, C)
        train_y = torch.from_numpy(train_y).view(-1 , 1)        # (N,) -> (N,1)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y).view(-1 , 1)

        # 2. 训练集随机划分20%验证集
        tr_x , val_x , tr_y , val_y = train_test_split(
            train_x , train_y,
            test_size=0.2,
            random_state=self.seed
        )

        # 3. 构建DataLoader
        train_loader = DataLoader(
            TensorDataset(tr_x , tr_y),
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False
        )
        valid_loader = DataLoader(
            TensorDataset(val_x , val_y),
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False
        )
        test_loader = DataLoader(
            TensorDataset(test_x , test_y),
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False
        )

        return train_loader , valid_loader , test_loader




        









