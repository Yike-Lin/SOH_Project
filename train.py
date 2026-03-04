import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from dataloader.xjtu_loader import XJTUDataset
from models.lstm import SOHLSTM
from utils.metrics import AverageMeter, eval_metrics



"""
从命令行获取超参数和配置
"""
def get_args():
    parser = argparse.ArgumentParser(description='SOH estimation with LSTM on XJTU')

    # 数据设置
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU'])
    parser.add_argument('--input_type', type=str, default='charge',
                        choices=['charge', 'partial_charge', 'handcraft_features'])
    parser.add_argument('--test_battery_id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--normalized_type', type=str, default='minmax',
                        choices=['minmax', 'standard'])
    parser.add_argument('--minmax_range', type=tuple, default=(-1, 1),
                        choices=[(0, 1), (-1, 1)])

    parser.add_argument('--batch', type=int, default=1,choices=[1,2,3,4,5,6,7,8,9])

    # 模型&训练设置
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_folder', default='results_LSTM')
    parser.add_argument('--experiment_num', type=int, default=1,help='同一配置重复实验次数')

    args = parser.parse_args()
    return args


"""
根据 args 加载数据
"""
def load_data(args):
    dataset = XJTUDataset(args)

    if args.input_type == 'charge':
        loaders = dataset.get_charge_data(test_battery_id = args.test_battery_id)
    elif args.input_type == 'partial_charge':
        laoders = dataset.get_partial_data(test_battery_id = args.test_battery_id)
    else:
        laoders = dataset.get_features(test_battery_id = args.test_battery_id)
        
    return loaders