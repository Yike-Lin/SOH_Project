import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, L, H)
        attn_scores = self.attention(x)          # (B, L, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * x, dim=1)  # (B, H)
        return context


class DualStreamMultiBiLSTMAttention(nn.Module):
    def __init__(self, input_channels=4, seq_len=128, hidden_size=128, num_layers=2):
        super(DualStreamMultiBiLSTMAttention, self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len
        bi_hidden_dim = hidden_size * 2

        # 充电流
        self.lstm_charge = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attn_charge = TemporalAttention(hidden_dim=bi_hidden_dim)

        # 放电流
        self.lstm_discharge = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attn_discharge = TemporalAttention(hidden_dim=bi_hidden_dim)

        # 融合 + 回归
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 4, 64),  # 2流 * (2向 * hidden_size)
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_charge, x_discharge):
        if x_charge.dim() == 3 and x_charge.shape[1] == self.input_channels:
            x_charge = x_charge.transpose(1, 2)
        if x_discharge.dim() == 3 and x_discharge.shape[1] == self.input_channels:
            x_discharge = x_discharge.transpose(1, 2)

        out_charge, _ = self.lstm_charge(x_charge)      # (B, L, 2H)
        feat_charge = self.attn_charge(out_charge)      # (B, 2H)

        out_discharge, _ = self.lstm_discharge(x_discharge)
        feat_discharge = self.attn_discharge(out_discharge)

        fused_features = torch.cat((feat_charge, feat_discharge), dim=1)  # (B, 4H)
        pred = self.regressor(fused_features)  # (B, 1)
        return pred