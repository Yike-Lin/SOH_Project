import torch
import torch.nn as nn

class SingleStreamBiLSTMAttention(nn.Module):
    """
    解耦后的单流特征提取器 (可分别用于充电流和放电流)
    """
    def __init__(self, input_channels=4, hidden_size=128, num_layers=2, use_bottleneck=True):
        super().__init__()
        # 双向 LSTM 提取时序特征
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # 时间注意力机制
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.use_bottleneck = use_bottleneck
        
        # 三板斧：使用 Bottleneck 提纯特征，并用 LayerNorm 抹平幅值偏移，GELU 激活保留梯度
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(hidden_size * 2, 128),
                nn.LayerNorm(128),
                nn.GELU()
            )
            self.regressor = nn.Linear(128, 1)
        else:
            self.regressor = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_len] -> 需要转置为 [Batch, Seq_len, Channels] 给 LSTM
        x = x.permute(0, 2, 1) 
        
        lstm_out, _ = self.lstm(x) # lstm_out: [Batch, Seq_len, Hidden*2]
        
        # Attention: 给 128 个时间步打分
        attn_weights = self.attn(lstm_out) # [Batch, Seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1) # 加权求和 -> [Batch, Hidden*2]
        
        if self.use_bottleneck:
            feat = self.bottleneck(context) # 纯净的 128 维老化特征
            soh = self.regressor(feat)
            return soh, feat
        else:
            soh = self.regressor(context)
            return soh, context