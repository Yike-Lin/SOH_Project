import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    时间注意力机制模块
    自动评估序列中不同时间步对预测 SOH 的重要性权重
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attn_scores = self.attention(x)  
        attn_weights = F.softmax(attn_scores, dim=1) 
        context = torch.sum(attn_weights * x, dim=1) 
        return context

class DualStreamMultiBiLSTMAttention(nn.Module):
    """
    双流 Multi-Bi-LSTM-Attention 网络
    流1: 充电数据 (Charge)
    流2: 放电数据 (Discharge)
    """
    def __init__(self, input_channels=4, seq_len=128, hidden_size=128, num_layers=2):
        super(DualStreamMultiBiLSTMAttention, self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len
        
        # 双向 LSTM 输出特征维度翻倍
        bi_hidden_dim = hidden_size * 2

        # 流 1: 充电数据特征提取
        self.lstm_charge = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 启用双向
        )
        self.attn_charge = TemporalAttention(hidden_dim=bi_hidden_dim)

        # 流 2: 放电数据特征提取
        self.lstm_discharge = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 启用双向
        )
        self.attn_discharge = TemporalAttention(hidden_dim=bi_hidden_dim)

        # 融合与回归网络
        # 将两股流的特征拼接，维度为: bi_hidden_dim + bi_hidden_dim = hidden_size * 4
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 4, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),  
            nn.Linear(64, 1)
        )

    def forward(self, x_charge, x_discharge):
        # 1. 形状校正 (N, C, L) -> (N, L, C)
        if x_charge.dim() == 3 and x_charge.shape[1] == self.input_channels:
            x_charge = x_charge.transpose(1, 2)
        if x_discharge.dim() == 3 and x_discharge.shape[1] == self.input_channels:
            x_discharge = x_discharge.transpose(1, 2)

        # 2. 充电侧前向传播
        out_charge, _ = self.lstm_charge(x_charge)
        feat_charge = self.attn_charge(out_charge)

        # 3. 放电侧前向传播
        out_discharge, _ = self.lstm_discharge(x_discharge)
        feat_discharge = self.attn_discharge(out_discharge)

        # 4. 双流特征拼接 (Fusion)
        fused_features = torch.cat((feat_charge, feat_discharge), dim=1)

        # 5. SOH 预测
        pred = self.regressor(fused_features)
        return pred

if __name__ == '__main__':
    x_c = torch.rand(30, 4, 128)
    x_d = torch.rand(30, 4, 128)
    net = DualStreamMultiBiLSTMAttention()
    y = net(x_c, x_d)
    print('Output shape:', y.shape)