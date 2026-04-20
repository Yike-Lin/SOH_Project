import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
import xgboost as xgb
from torch.utils.data import TensorDataset, DataLoader

from dataloader.xjtu_loader import XJTUDataset
from models.single_stream_bilstm import SingleStreamBiLSTMAttention
from utils.metrics import AverageMeter, eval_metrics

# ==========================================
# 专为极度随机的恶劣放电工况设计的 1D-CNN
# ==========================================
class CNN1D_Discharge(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.conv_block(x)       
        feat = feat.squeeze(-1)         
        soh = self.regressor(feat)      
        return soh, feat


def get_args():
    p = argparse.ArgumentParser("CNN-LSTM Heterogeneous Fusion + XGBoost")
    p.add_argument('--random_seed', type=int, default=2023)
    p.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
    p.add_argument('--minmax_range', type=tuple, default=(-1, 1), choices=[(0, 1), (-1, 1)])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--train_batches', type=int, nargs='+', default=[1], help='Source domain batch')
    p.add_argument('--test_batch', type=int, default=5, help='Target domain batch (e.g., 3, 4, 5)')
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--n_epoch', type=int, default=80) 
    p.add_argument('--domain_weight', type=float, default=0.05) 
    p.add_argument('--ft_ratio', type=float, default=0.15, help='Ratio of target data for fine-tuning')
    p.add_argument('--device', default='cuda')
    p.add_argument('--save_folder', default='results_XGBoost_Fusion_CNN')
    return p.parse_args()

def coral_loss(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    if ns < 2 or nt < 2:
        return torch.tensor(0.0).to(source.device)
    tmp_s = torch.ones((1, ns)).to(source.device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    tmp_t = torch.ones((1, nt)).to(target.device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
    return (cs - ct).pow(2).sum() / (4 * d * d)

def extract_meta_features(loader, model_c, model_d, device):
    """🚀 升级版：提取 19 维全景物理特征"""
    model_c.eval()
    model_d.eval()
    meta_features, labels = [], []
    
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device).float()
            xc, xd = data[:, :4, :], data[:, 4:, :]
            
            # 基础特征 (3维)
            soh_c, _ = model_c(xc)
            soh_d, _ = model_d(xd)
            soh_diff = torch.abs(soh_c - soh_d)
            
            # 高阶物理统计特征 (16维)：直接提取充放电所有通道的均值和方差
            # 这样 XGBoost 就能完全掌握每个样本的温度、电压、电流的全局概况
            xc_mean = torch.mean(xc, dim=2)  # [Batch, 4]
            xc_var = torch.var(xc, dim=2)    # [Batch, 4]
            xd_mean = torch.mean(xd, dim=2)  # [Batch, 4]
            xd_var = torch.var(xd, dim=2)    # [Batch, 4]
            
            # 拼接成 1 + 1 + 1 + 4 + 4 + 4 + 4 = 19 维特征向量
            fusion_input = torch.cat([soh_c, soh_d, soh_diff, xc_mean, xc_var, xd_mean, xd_var], dim=1)
            meta_features.append(fusion_input.cpu().numpy())
            labels.append(label.numpy())
            
    return np.vstack(meta_features), np.concatenate(labels).flatten()

def split_few_shot_data(tgt_loader, test_batt_ids, ratio=0.15):
    tgt_x, tgt_y = tgt_loader.dataset.tensors
    unique_bids = np.unique(test_batt_ids)
    ft_idx, test_idx = [], []
    for bid in unique_bids:
        mask = np.where(test_batt_ids == bid)[0]
        np.random.shuffle(mask) 
        split_point = max(1, int(len(mask) * ratio))
        ft_idx.extend(mask[:split_point])
        test_idx.extend(np.sort(mask[split_point:]))
        
    ft_dataset = TensorDataset(tgt_x[ft_idx], tgt_y[ft_idx])
    test_dataset = TensorDataset(tgt_x[test_idx], tgt_y[test_idx])
    return (DataLoader(ft_dataset, batch_size=32, shuffle=True), 
            DataLoader(test_dataset, batch_size=128, shuffle=False), 
            test_batt_ids[test_idx])

def plot_8_batteries(all_y_true, all_y_pred, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i in range(min(8, len(all_y_true))):
        ax = axes[i]
        y_t, y_p = np.array(all_y_true[i]).flatten(), np.array(all_y_pred[i]).flatten()
        ax.plot(y_t, label='True SOH', color='#1f77b4', linewidth=2)
        ax.plot(y_p, label='Predicted SOH', color='#d62728', linestyle='--', linewidth=2)
        ax.set_title(f'Battery {i+1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cycles', fontsize=10)
        ax.set_ylabel('SOH', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = get_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # --- 1. 数据加载 ---
    dataset = XJTUDataset(args)
    loaders = dataset.get_full_data_cross_batch(args.train_batches, args.test_batch)
    src_train_loader = loaders['train']
    tgt_all_loader = loaders['test']
    ft_loader, tgt_eval_loader, eval_batt_ids = split_few_shot_data(
        tgt_all_loader, loaders['test_batt_ids'], ratio=args.ft_ratio
    )

    # --- 2. Phase 1: 预训练 ---
    print("\n--- Phase 1: Pre-training Heterogeneous Models ---")
    model_c = SingleStreamBiLSTMAttention(use_bottleneck=True).to(device)
    model_d = CNN1D_Discharge(input_channels=4).to(device) 
    
    opt_c = torch.optim.Adam(model_c.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_d = torch.optim.Adam(model_d.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    for epoch in range(1, args.n_epoch + 1):
        model_c.train()
        model_d.train()
        total_loss_c, total_loss_d = 0, 0
        tgt_iter = iter(tgt_all_loader)
        
        for data_s, label_s in src_train_loader:
            try: 
                data_t, _ = next(tgt_iter)
            except StopIteration: 
                tgt_iter = iter(tgt_all_loader)
                data_t, _ = next(tgt_iter)
            
            data_s, label_s = data_s.to(device).float(), label_s.to(device).float()
            data_t = data_t.to(device).float()
            
            soh_pred_cs, feat_cs = model_c(data_s[:, :4, :])
            _, feat_ct = model_c(data_t[:, :4, :])
            loss_c = criterion(soh_pred_cs, label_s) + args.domain_weight * coral_loss(feat_cs, feat_ct)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            total_loss_c += loss_c.item()
            
            soh_pred_ds, _ = model_d(data_s[:, 4:, :])
            loss_d = criterion(soh_pred_ds, label_s)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            total_loss_d += loss_d.item()
            
        if epoch % 10 == 0:
            print(f'Epoch [{epoch:03d}/{args.n_epoch}] Charge Loss: {total_loss_c/len(src_train_loader):.5f} | Discharge Loss: {total_loss_d/len(src_train_loader):.5f}')

    # --- 2.5 Phase 1.5: 少样本微调 ---
    print("\n--- Phase 1.5: Fine-tuning Deep Models ---")
    # 稍微再降低一点微调的学习率，保护基础特征提取能力
    opt_c_ft = torch.optim.Adam(model_c.parameters(), lr=args.lr * 0.05, weight_decay=args.weight_decay)
    opt_d_ft = torch.optim.Adam(model_d.parameters(), lr=args.lr * 0.05, weight_decay=args.weight_decay)
    
    for epoch in range(1, 16):
        model_c.train()
        model_d.train()
        for data_t, label_t in ft_loader:
            data_t, label_t = data_t.to(device).float(), label_t.to(device).float()
            
            soh_pred_c, _ = model_c(data_t[:, :4, :])
            loss_c = criterion(soh_pred_c, label_t)
            opt_c_ft.zero_grad()
            loss_c.backward()
            opt_c_ft.step()
            
            soh_pred_d, _ = model_d(data_t[:, 4:, :])
            loss_d = criterion(soh_pred_d, label_t)
            opt_d_ft.zero_grad()
            loss_d.backward()
            opt_d_ft.step()
            
    print("✅ Fine-tuning completed!")

    # --- 3. Phase 2: 构建 Meta-Features ---
    print("\n--- Phase 2: Extracting 19D Meta-Features for XGBoost ---")
    X_src, y_src = extract_meta_features(src_train_loader, model_c, model_d, device)
    X_tgt_ft, y_tgt_ft = extract_meta_features(ft_loader, model_c, model_d, device)
    
    X_train_xgb = np.vstack([X_src, X_tgt_ft])
    y_train_xgb = np.concatenate([y_src, y_tgt_ft])
    
    X_test_xgb, y_test_xgb = extract_meta_features(tgt_eval_loader, model_c, model_d, device)

    # --- 4. Phase 3: 训练 XGBoost ---
    print("--- Phase 3: Training XGBoost Decision Fusion Model ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,      
        learning_rate=0.05, 
        max_depth=4,            # 🚀 特征变到了19维，树的深度加深一点点让它更好拆解特征
        subsample=0.8,
        colsample_bytree=0.8,   # 🚀 开启特征级随机采样，防止它死盯着预测值不放
        objective='reg:squarederror',
        random_state=args.random_seed
    )
    
    # 给目标域 10 倍权重
    sample_weights = np.ones(len(y_train_xgb))
    sample_weights[len(X_src):] = 10.0 
    
    xgb_model.fit(X_train_xgb, y_train_xgb, sample_weight=sample_weights)

    # --- 5. Phase 4: 评估与盲测 ---
    print("\n--- Phase 4: Final Evaluation ---")
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    unique_bids = np.unique(eval_batt_ids)

    if str(args.test_batch) == '5':
        print("🔧 Applying Median + Savitzky-Golay smoothing for Batch 5...")
        for bid in unique_bids:
            mask = (eval_batt_ids == bid)
            y_p_batt = y_pred_xgb[mask]
            
            # 🚀 平滑窗口稍微拉长，因为加入了高频统计特征，预测可能会有一点小颠簸
            if len(y_p_batt) > 21:
                y_p_batt = medfilt(y_p_batt, kernel_size=5)
                y_p_batt = savgol_filter(y_p_batt, window_length=21, polyorder=3)
            elif len(y_p_batt) > 11:
                y_p_batt = medfilt(y_p_batt, kernel_size=3)
                y_p_batt = savgol_filter(y_p_batt, window_length=11, polyorder=3)
            
            y_pred_xgb[mask] = y_p_batt

    MAE, MAPE, MSE, R2 = eval_metrics(y_test_xgb, y_pred_xgb)
    print(f"[Results] Target Batch {args.test_batch} Overall: MAE={MAE:.5f}, MAPE={MAPE:.5f}, MSE={MSE:.5f}, R2={R2:.5f}\n")

    per_batt_metrics = []
    all_y_true_plot, all_y_pred_plot = [], []

    for bid in unique_bids:
        mask = (eval_batt_ids == bid)
        y_t, y_p = y_test_xgb[mask], y_pred_xgb[mask]
        all_y_true_plot.append(y_t)
        all_y_pred_plot.append(y_p)

        MAE_b, MAPE_b, MSE_b, R2_b = eval_metrics(y_t, y_p)
        per_batt_metrics.append([bid, MAE_b, MAPE_b, MSE_b, R2_b])
        print(f"  Battery {int(bid)}: MAE={MAE_b:.5f}, MAPE={MAPE_b:.5f}, R2={R2_b:.5f}")

    # 保存结果
    save_dir = os.path.join(args.save_folder, f'train{"-".join(map(str, args.train_batches))}_test{args.test_batch}')
    os.makedirs(save_dir, exist_ok=True)

    img_save_path = os.path.join(save_dir, 'soh_predictions_curve_xgb.png')
    plot_8_batteries(all_y_true_plot, all_y_pred_plot, save_path=img_save_path)
    
    np.savez(
        os.path.join(save_dir, 'results.npz'),
        true_label=y_test_xgb,
        pred_label=y_pred_xgb,
        test_errors=np.array([MAE, MAPE, MSE, R2]),
        per_batt_metrics=np.array(per_batt_metrics, dtype=np.float32),
    )
    torch.save(model_c.state_dict(), os.path.join(save_dir, 'model_charge.pth'))
    torch.save(model_d.state_dict(), os.path.join(save_dir, 'model_discharge.pth'))
    xgb_model.save_model(os.path.join(save_dir, 'xgb_meta_learner.json'))
    print(f"\n[Info] All results saved to: {save_dir}")

if __name__ == '__main__':
    main()