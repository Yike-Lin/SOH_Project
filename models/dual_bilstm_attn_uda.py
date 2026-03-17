import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        
        # 保留关键的末端时间步残差直连
        last_state = x[:, -1, :]
        out = torch.cat([context, last_state], dim=1)
        return out


class DualStreamBiLSTMAttentionUDA(nn.Module):
    """
    非对称跨域对齐 (Charge-Only UDA)
    剥离了对归一化敏感的 Wide 分支，全权交由深度网络进行特征融合
    """
    def __init__(
        self,
        input_channels=4,
        seq_len=128,
        hidden_size=128,
        num_layers=2,
        num_domains=2
    ):
        super(DualStreamBiLSTMAttentionUDA, self).__init__()

        self.input_channels = input_channels
        self.bi_hidden = hidden_size * 2
        self.attn_out_dim = self.bi_hidden * 2 
        fuse_dim = self.attn_out_dim * 2  

        # ================= 流 1：充电 =================
        self.lstm_charge = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.attn_charge = TemporalAttention(self.bi_hidden)

        # ================= 流 2：放电 =================
        self.lstm_discharge = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.attn_discharge = TemporalAttention(self.bi_hidden)

        # ================= 回归预测头 =================
        self.regressor = nn.Sequential(
            nn.Linear(fuse_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # ================= 域分类头 =================
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.attn_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_domains)
        )

    def forward(self, x_charge, x_discharge, alpha=0.0):
        if x_charge.dim() == 3 and x_charge.size(1) == self.input_channels:
            x_charge = x_charge.transpose(1, 2)
        if x_discharge.dim() == 3 and x_discharge.size(1) == self.input_channels:
            x_discharge = x_discharge.transpose(1, 2)

        out_c, _ = self.lstm_charge(x_charge)   
        feat_c = self.attn_charge(out_c)        

        out_d, _ = self.lstm_discharge(x_discharge)
        feat_d = self.attn_discharge(out_d)

        # 核心：只让 feat_c 走 GRL 对抗
        if alpha > 0:
            rev_feat_c = grad_reverse(feat_c, lambd=alpha)
            domain_logits = self.domain_classifier(rev_feat_c)
        else:
            domain_logits = None

        # 拼接进行预测 (无偏差的深度回归)
        fused = torch.cat([feat_c, feat_d], dim=1) 
        soh_pred = self.regressor(fused)

        return soh_pred, domain_logits, fused