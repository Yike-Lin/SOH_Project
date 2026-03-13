import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dataloader.xjtu_loader import XJTUDataset
from models.dual_bilstm_attn_uda import DualStreamBiLSTMAttentionUDA
from utils.metrics import AverageMeter, eval_metrics


def get_args():
    p = argparse.ArgumentParser("Cross-batch Few-Shot Fine-Tuning on XJTU full data")

    # 数据设置
    p.add_argument('--random_seed', type=int, default=2023)
    p.add_argument('--normalized_type', type=str, default='minmax',
                   choices=['minmax', 'standard'])
    p.add_argument('--minmax_range', type=tuple, default=(-1, 1),
                   choices=[(0, 1), (-1, 1)])
    p.add_argument('--batch_size', type=int, default=128)

    # 跨 batch
    p.add_argument('--train_batches', type=int, nargs='+', default=[1],
                   help='源域使用哪些 batch，例如: --train_batches 1 2')
    p.add_argument('--test_batch', type=int, default=3,
                   help='目标域使用哪个 batch，例如: --test_batch 3')

    # 模型&训练 (Pre-train)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--n_epoch', type=int, default=100)
    p.add_argument('--early_stop', type=int, default=20)
    
    # 微调设置 (Fine-tune)
    p.add_argument('--ft_ratio', type=float, default=0.15, help='目标域用于微调的数据比例')
    p.add_argument('--ft_epoch', type=int, default=40, help='微调的 epoch 数量')
    p.add_argument('--ft_lr', type=float, default=1e-4, help='微调的极小学习率')

    p.add_argument('--device', default='cuda')
    p.add_argument('--save_folder', default='results_FewShot_cross_batch')

    args = p.parse_args()
    return args


def evaluate_soh(model, loader, device):
    """
    只做 SOH 推理，用于在源域 valid 或目标域 test 上评估回归性能
    """
    model.eval()
    meter = AverageMeter()
    true_list, pred_list = [], []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device).float()
            label = label.to(device).float()

            xc = data[:, :4, :]
            xd = data[:, 4:, :]

            soh_pred, _ = model(xc, xd, alpha=0.0)
            loss = criterion(soh_pred, label)
            meter.update(loss.item(), n=data.size(0))

            true_list.append(label.cpu().numpy())
            pred_list.append(soh_pred.cpu().numpy())

    y_true = np.concatenate(true_list, axis=0)
    y_pred = np.concatenate(pred_list, axis=0)
    return meter.avg, y_true, y_pred


def split_few_shot_data(tgt_loader, test_batt_ids, ratio=0.15):
    """
    终极改进：从目标域数据中，每块电池【随机】抽取 ratio (比如 15%) 的周期用于微调，
    剩下的用于严格测试。覆盖完整的 SOH 衰减区间！
    """
    tgt_x, tgt_y = tgt_loader.dataset.tensors
    unique_bids = np.unique(test_batt_ids)
    
    ft_idx, test_idx = [], []
    for bid in unique_bids:
        mask = np.where(test_batt_ids == bid)[0]
        # 核心魔法：随机打乱索引
        np.random.shuffle(mask)
        # 按比例切分 (至少保证有1个样本)
        split_point = max(1, int(len(mask) * ratio))
        
        ft_idx.extend(mask[:split_point])
        test_idx.extend(mask[split_point:])
        
    ft_dataset = TensorDataset(tgt_x[ft_idx], tgt_y[ft_idx])
    test_dataset = TensorDataset(tgt_x[test_idx], tgt_y[test_idx])
    
    # 微调时 batch_size 设小一点，充分更新
    ft_loader = DataLoader(ft_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return ft_loader, eval_loader, test_batt_ids[test_idx]


def main():
    args = get_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(f'Source(train) batches: {args.train_batches}, Target(test) batch: {args.test_batch}')
    
    # 1) 加载数据
    dataset = XJTUDataset(args)
    loaders = dataset.get_full_data_cross_batch(
        train_batches=args.train_batches,
        test_batch=args.test_batch
    )
    src_train_loader = loaders['train']
    src_valid_loader = loaders['valid']
    tgt_all_loader = loaders['test']
    test_batt_ids = loaders['test_batt_ids']

    # ================= 关键步骤 1：构建少样本数据集 =================
    ft_loader, tgt_eval_loader, eval_batt_ids = split_few_shot_data(
        tgt_all_loader, test_batt_ids, ratio=args.ft_ratio
    )
    print(f"Target Domain Split (Ratio={args.ft_ratio}): Fine-Tune set={len(ft_loader.dataset)}, Test set={len(tgt_eval_loader.dataset)}")

    x0, _ = next(iter(src_train_loader))
    C, L = x0.size(1), x0.size(2)

    model = DualStreamBiLSTMAttentionUDA(
        input_channels=4, seq_len=L, hidden_size=128, num_layers=2, num_domains=2
    ).to(device)

    # ================= 关键步骤 2：在源域上正常预训练 =================
    print("\n--- Phase 1: Pre-training on Source Domain ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    best_valid = float('inf')
    best_state = None
    early_counter = 0
    train_losses, valid_losses = [], []
    
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        total_loss = 0
        for data, label in src_train_loader:
            data, label = data.to(device).float(), label.to(device).float()
            xc, xd = data[:, :4, :], data[:, 4:, :]
            # 不用 alpha (即不使用 GRL)，纯监督预训练
            soh_pred, _ = model(xc, xd, alpha=0.0)
            loss = criterion(soh_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            
        avg_train_loss = total_loss / len(src_train_loader.dataset)
        valid_loss, _, _ = evaluate_soh(model, src_valid_loader, device)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(valid_loss)
        
        print(f'Pretrain Epoch [{epoch}/{args.n_epoch}] Train Loss: {avg_train_loss:.5f} | Valid Loss: {valid_loss:.5f}')
        
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = model.state_dict()
            early_counter = 0
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                print("Pre-training early stopped.")
                break

    # 载入最好的预训练权重
    model.load_state_dict(best_state)

    # ================= 关键步骤 3：在目标域进行全局微调 =================
    print("\n--- Phase 2: Global Fine-Tuning on Target Domain ---")
    
    # 确保所有参数都解冻，让 LSTM 也稍微适应目标域的波形
    for param in model.parameters():
        param.requires_grad = True
            
    # 使用极小的学习率进行温柔的全局微调
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr)
    
    for epoch in range(1, args.ft_epoch + 1):
        model.train()
        total_loss = 0
        for data, label in ft_loader:
            data, label = data.to(device).float(), label.to(device).float()
            xc, xd = data[:, :4, :], data[:, 4:, :]
            soh_pred, _ = model(xc, xd, alpha=0.0)
            loss = criterion(soh_pred, label)
            
            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
            total_loss += loss.item() * data.size(0)
            
        if epoch % 5 == 0 or epoch == 1:
            avg_loss = total_loss / len(ft_loader.dataset)
            print(f"Fine-Tune Epoch [{epoch}/{args.ft_epoch}] Loss: {avg_loss:.6f}")
            
    # ================= 关键步骤 4：在目标域进行最终评估 =================
    print("\n--- Phase 3: Final Evaluation on Target Domain ---")
    _, y_true, y_pred = evaluate_soh(model, tgt_eval_loader, device)
    
    MAE, MAPE, MSE, R2 = eval_metrics(y_true, y_pred)
    print(f'[Final Fine-Tuned] Test on batch{args.test_batch} (excluding {args.ft_ratio*100}% random cycles): ')
    print(f'MAE={MAE:.5f}, MAPE={MAPE:.5f}, MSE={MSE:.5f}, R2={R2:.5f}')

    # 按电池 ID 评估
    unique_bids = np.unique(eval_batt_ids)
    per_batt_metrics = []
    for bid in unique_bids:
        mask = (eval_batt_ids == bid)
        y_t, y_p = y_true[mask], y_pred[mask]
        MAE_b, MAPE_b, MSE_b, R2_b = eval_metrics(y_t, y_p)
        per_batt_metrics.append([bid, MAE_b, MAPE_b, MSE_b, R2_b])
        print(f'  Battery {int(bid)}: MAE={MAE_b:.5f}, MAPE={MAPE_b:.5f}, MSE={MSE_b:.5f}, R2={R2_b:.5f}')

    # ================= 保存结果 =================
    per_batt_metrics = np.array(per_batt_metrics, dtype=np.float32)
    save_dir = os.path.join(
        args.save_folder,
        f'train{"-".join(map(str, args.train_batches))}',
        f'test{args.test_batch}'
    )
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, 'results.npz'),
        train_loss=np.array(train_losses),
        valid_loss=np.array(valid_losses),
        true_label=y_true,
        pred_label=y_pred,
        test_errors=np.array([MAE, MAPE, MSE, R2]),
        per_batt_metrics=per_batt_metrics,
    )
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_ft_model.pth'))
    print('\nResults saved to:', save_dir)


if __name__ == '__main__':
    main()