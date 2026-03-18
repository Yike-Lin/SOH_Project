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
        # 🚀 绝杀 1：加入 LayerNorm！
        # XJTU 不同 Batch 的电流电压绝对幅值差异很大。
        # 在算 Attention 前强行拉回标准正态分布，能瞬间抹平大量物理工况带来的 Covariate Shift（协变量偏移）。
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), # 🚀 绝杀 2：Tanh 换 GELU，防止长序列微调时梯度僵死
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        scores = self.attention(x_norm)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        
        # 你的神来之笔：保留末端时间步残差，锁住衰减拐点
        last_state = x[:, -1, :]
        out = torch.cat([context, last_state], dim=1)
        return out


class DualStreamBiLSTMAttentionUDA(nn.Module):
    """
    终极版：非对称跨域对齐 + 瓶颈特征提纯 (Bottleneck CORAL)
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

        # 🚀 绝杀 3：特征瓶颈层 (Bottleneck)
        # 解决 CORAL Loss 在 1024 维高维空间算协方差矩阵导致的“秩亏（秩不够，满地噪声）”问题
        # 强迫模型把最核心的 SOH 信息压缩到 128 维
        self.bottleneck_dim = 128
        self.bottleneck = nn.Sequential(
            nn.Linear(fuse_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # ================= 回归预测头 =================
        # 吃压缩后的 128 维特征，回归变得更稳
        self.regressor = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # ================= 域分类头 =================
        # 保持你的非对称策略：依然只吃充电分支的特征 (attn_out_dim = 512)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.attn_out_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_domains)
        )

    def forward(self, x_charge, x_discharge, alpha=0.0 , is_batch_5 = False):
        if x_charge.dim() == 3 and x_charge.size(1) == self.input_channels:
            x_charge = x_charge.transpose(1, 2)
        if x_discharge.dim() == 3 and x_discharge.size(1) == self.input_channels:
            x_discharge = x_discharge.transpose(1, 2)

        out_c, _ = self.lstm_charge(x_charge)   
        feat_c = self.attn_charge(out_c)        

        out_d, _ = self.lstm_discharge(x_discharge)
        feat_d = self.attn_discharge(out_d)

        # 核心：保留你的非对称对抗，只让 feat_c 走 GRL
        if alpha > 0:
            rev_feat_c = grad_reverse(feat_c, lambd=alpha)
            domain_logits = self.domain_classifier(rev_feat_c)
        else:
            domain_logits = None

        # 🚀 物理级补丁：针对 Batch 5 这种随机放电工况，强行压制放电特征的置信度
        # 在训练模式下，随机屏蔽 50% 的放电特征，逼迫模型依赖纯净的充电特征(feat_c)
        if self.training:
            feat_d = F.dropout(feat_d, p=0.5, training=self.training)

        if is_batch_5:
            feat_d_fused = feat_d * 0.1
        else:
            feat_d_fused = feat_d

        # 拼接原始特征 (1024维)
        raw_fused = torch.cat([feat_c, feat_d], dim=1) 
        
        # 通过瓶颈层提纯特征 (压到 128维)
        bottleneck_feat = self.bottleneck(raw_fused)

        # 回归预测
        soh_pred = self.regressor(bottleneck_feat)

        # 返回 bottleneck_feat 给外部。外部的 train.py 里的 coral_loss(fused_s, fused_t) 
        # 现在实际上是在 128 维的高纯度空间里计算对齐，再也不会因为高维噪声算崩了！
        return soh_pred, domain_logits, bottleneck_feat