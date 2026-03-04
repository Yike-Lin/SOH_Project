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
        self.root = 'data/XJTU'
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
    

    """
    从mat文件读出所有电池,留一电池test
    """
    def _get_raw_data(self , path , test_battery_id):
        # 1. 读取文件与校验
        mat = loadmat(path)
        battery = mat['battery']

        num_batt = battery.shape[1]
        battery_ids = list(range(1 , num_batt + 1))

        if test_battery_id not in battery_ids:
            raise IndentationError(f'test_battery_id must be in {battery_ids} , got {test_battery_id}')

        # 2. 提取测试集数据
        test_battery = battery[0 , test_battery_id -1][0]
        test_x , test_y = self._parser_mat_data(test_battery)

        # 3. 提取并合并训练集数据
        train_x_list , train_y_list = [],[]
        for bid in battery_ids:
            if bid == test_battery_id:
                continue
            train_battery = battery[0 , bid -1][0]
            x , y = self._parser_mat_data(train_battery)
            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.concatenate(train_x_list , axis=0)
        train_y = np.concatenate(train_y_list , axis=0)

        # 4. 封装 返回
        return self._encapsulation(train_x , train_y , test_x , test_y)

    
    """
    加载完整充电曲线数据接口
    """
    def get_charge_data(self , test_battery_id = 1):
        file_name = f'batch-{self.batch}.mat'
        path = os.path.join(self.root , 'charge' , file_name)

        train_loader , valid_laoder , test_loader = self._get_raw_data(path , test_battery_id)
        return {'train': train_loader , 'valid': valid_laoder , 'test': test_loader}
    
    """
    加载部分充电曲线数据接口
    """
    def get_partial_data(self , test_battery_id = 1):
        if self.batch == 6:
            file_name = f'batch-{self.batch}_3.9-4.19.mat'
        else:
            file_name = f'batch-{self.batch}_3.7-4.1.mat'

        path = os.path.join(self.root, 'partial_charge', file_name)
        train_loader, valid_loader, test_loader = self._get_raw_data(path, test_battery_id)
        return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    """
    解析电池在 Excel 表中的特征数据
    """
    def _parser_xlsx(self , df_i: pd.DataFrame):
        x = np.asarray(df_i.iloc[:, :-1], dtype=np.float32)
        label = np.asarray(df_i['label'], dtype=np.float32)
        x = self._normalize(x)
        soh = label / self.max_capacity
        return x, soh
    
    """
    加载手工特征数据
    """
    def get_features(self , test_battery_id = 1):
        file_name = f'batch-{self.batch}_features.xlsx'
        path = os.path.join(self.root, 'handcraft_features', file_name)

        df_dict = pd.read_excel(path, sheet_name=None)
        sheet_names = list(df_dict.keys())
        battery_ids = list(range(1, len(sheet_names) + 1))

        if test_battery_id not in battery_ids:
            raise IndexError(f'test_battery_id must be in {battery_ids}, got {test_battery_id}')

        test_df = pd.read_excel(path, sheet_name=test_battery_id - 1, header=0)
        test_x, test_y = self._parser_xlsx(test_df)

        train_x_list, train_y_list = [], []
        for bid in battery_ids:
            if bid == test_battery_id:
                continue
            df_i = df_dict[sheet_names[bid - 1]]
            x, y = self._parser_xlsx(df_i)
            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        return self._encapsulation(train_x, train_y, test_x, test_y)
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--normalized_type', type=str, default='minmax')
    parser.add_argument('--minmax_range', type=tuple, default=(-1, 1))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args([])  # [] 表示不用命令行参数，使用默认值

    dataset = XJTUDataset(args)
    loaders = dataset.get_charge_data(test_battery_id=1)

    for split in ['train', 'valid', 'test']:
        dl = loaders[split]
        x, y = next(iter(dl))
        print(split, 'batch x shape:', x.shape, 'batch y shape:', y.shape)
        




        









