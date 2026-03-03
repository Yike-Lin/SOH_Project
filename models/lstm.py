import torch
import torch.nn as nn


class SOHLSTM(nn.Module):

    """
    配置模型
    """
    def __init__(self , input_channels = 4 , seq_len = 128 , hidden_size = 128 , num_layers = 2):
        super(SOHLSTM , self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len

        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size = input_channels,    # 输入通道数
            hidden_size = hidden_size,      # 序列长度  
            num_layers = num_layers,        # LSTM隐层维度
            batch_first = True              # 数据排列方式改为(样本, 时间, 特征)
        )

        # 定义最后的回归网络，把最后一步的隐状态映射到标量(SOH)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size , 64),    # 降维组合
            nn.LeakyReLU(),                 # 加入激活函数
            nn.Linear(64 , 1)               # 合并输出
        )



        
