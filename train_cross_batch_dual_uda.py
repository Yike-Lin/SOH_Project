import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from dataloader.xjtu_loader import XJTUDataset
from models.dual_bilstm_attn_uda import DualStreamBiLSTMAttentionUDA
from utils.metrics import AverageMeter, eval_metrics


def get_args():
    p = argparse.ArgumentParser("Cross-batch SSDA (UDA + Few-Shot) on XJTU")
    
    # Dataset Config
    p.add_argument('--random_seed', type=int, default=2023)
    p.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
    p.add_argument('--minmax_range', type=tuple, default=(-1, 1), choices=[(0, 1), (-1, 1)])
    p.add_argument('--batch_size', type=int, default=128)

    # Cross-batch Config
    p.add_argument('--train_batches', type=int, nargs='+', default=[1], help='Source domain batch')
    p.add_argument('--test_batch', type=int, default=3, help='Target domain batch')

    # Phase 1: UDA Config
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--n_epoch', type=int, default=100)
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--domain_weight', type=float, default=0.05)
    
    # Phase 2: Few-Shot Fine-Tuning Config
    p.add_argument('--ft_ratio', type=float, default=0.15, help='Ratio of target data for fine-tuning')
    p.add_argument('--ft_epoch', type=int, default=50)
    p.add_argument('--ft_lr', type=float, default=5e-4)

    p.add_argument('--device', default='cuda')
    p.add_argument('--save_folder', default='results_SSDA_cross_batch')

    return p.parse_args()


def evaluate_soh(model, loader, device):
    """Inference for SOH prediction only (domain adaptation disabled)"""
    model.eval()
    meter = AverageMeter()
    true_list, pred_list = [], []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device).float(), label.to(device).float()
            xc, xd = data[:, :4, :], data[:, 4:, :]

            soh_pred, _ = model(xc, xd, alpha=0.0)
            loss = criterion(soh_pred, label)
            meter.update(loss.item(), n=data.size(0))

            true_list.append(label.cpu().numpy())
            pred_list.append(soh_pred.cpu().numpy())

    y_true = np.concatenate(true_list, axis=0)
    y_pred = np.concatenate(pred_list, axis=0)
    return meter.avg, y_true, y_pred


def split_few_shot_data(tgt_loader, test_batt_ids, ratio=0.15):
    """Split target domain: random sampling for fine-tuning, sorted temporal data for testing."""
    tgt_x, tgt_y = tgt_loader.dataset.tensors
    unique_bids = np.unique(test_batt_ids)
    
    ft_idx, test_idx = [], []
    for bid in unique_bids:
        mask = np.where(test_batt_ids == bid)[0]
        np.random.shuffle(mask) 
        
        split_point = max(1, int(len(mask) * ratio))
        ft_idx.extend(mask[:split_point])
        
        # Sort test indices to maintain temporal degradation curve for visualization
        test_idx.extend(np.sort(mask[split_point:]))
        
    ft_dataset = TensorDataset(tgt_x[ft_idx], tgt_y[ft_idx])
    test_dataset = TensorDataset(tgt_x[test_idx], tgt_y[test_idx])
    
    ft_loader = DataLoader(ft_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    eval_batt_ids = test_batt_ids[test_idx]
    
    return ft_loader, eval_loader, eval_batt_ids


def plot_8_batteries(all_y_true, all_y_pred, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i in range(len(all_y_true)):
        ax = axes[i]
        y_t = np.array(all_y_true[i]).flatten()
        y_p = np.array(all_y_pred[i]).flatten()
        ax.plot(y_t, label='True SOH', color='#1f77b4', linewidth=2)
        ax.plot(y_p, label='Predicted SOH', color='#d62728', linestyle='--', linewidth=2)
        ax.set_title(f'Battery {i+1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cycles', fontsize=10)
        ax.set_ylabel('SOH', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim([min(y_t.min(), y_p.min()) - 0.02, max(y_t.max(), y_p.max()) + 0.02])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] SOH curves saved to {os.path.abspath(save_path)}")
    plt.close()


def main():
    args = get_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load Data
    dataset = XJTUDataset(args)
    loaders = dataset.get_full_data_cross_batch(
        train_batches=args.train_batches,
        test_batch=args.test_batch
    )
    src_train_loader = loaders['train']
    src_valid_loader = loaders['valid']
    tgt_all_loader = loaders['test']
    test_batt_ids = loaders['test_batt_ids']

    ft_loader, tgt_eval_loader, eval_batt_ids = split_few_shot_data(
        tgt_all_loader, test_batt_ids, ratio=args.ft_ratio
    )
    print(f"[Data] Source Domain: {len(src_train_loader.dataset)} samples")
    print(f"[Data] Target Domain: FT ({int(args.ft_ratio*100)}%) = {len(ft_loader.dataset)} | Test = {len(tgt_eval_loader.dataset)}\n")

    x0, _ = next(iter(src_train_loader))
    model = DualStreamBiLSTMAttentionUDA(
        input_channels=4, seq_len=x0.size(2), hidden_size=128, num_layers=2, num_domains=2
    ).to(device)

    # --- Phase 1: UDA Pre-training ---
    print("--- Phase 1: UDA Pre-training ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss() 
    
    best_valid = float('inf')
    best_state = None
    early_counter = 0
    
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        total_loss = 0
        tgt_iter = iter(tgt_all_loader)
        
        for i, (data_s, label_s) in enumerate(src_train_loader):
            try:
                data_t, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_all_loader)
                data_t, _ = next(tgt_iter)
                
            data_s, label_s = data_s.to(device).float(), label_s.to(device).float()
            data_t = data_t.to(device).float()
            
            xc_s, xd_s = data_s[:, :4, :], data_s[:, 4:, :]
            xc_t, xd_t = data_t[:, :4, :], data_t[:, 4:, :]
            
            p = float(i + (epoch - 1) * len(src_train_loader)) / (args.n_epoch * len(src_train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            soh_pred_s, domain_pred_s = model(xc_s, xd_s, alpha=alpha)
            loss_soh = criterion(soh_pred_s, label_s)
            _, domain_pred_t = model(xc_t, xd_t, alpha=alpha)
            
            if domain_pred_s is not None and domain_pred_t is not None:
                domain_label_s = torch.zeros(data_s.size(0), dtype=torch.long).to(device)
                domain_label_t = torch.ones(data_t.size(0), dtype=torch.long).to(device)
                
                loss_domain_s = domain_criterion(domain_pred_s, domain_label_s)
                loss_domain_t = domain_criterion(domain_pred_t, domain_label_t)
                loss = loss_soh + args.domain_weight * (loss_domain_s + loss_domain_t)
            else:
                loss = loss_soh
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data_s.size(0)
            
        avg_train_loss = total_loss / len(src_train_loader.dataset)
        valid_loss, _, _ = evaluate_soh(model, src_valid_loader, device)
        
        print(f'Epoch [{epoch:03d}/{args.n_epoch}] Train Loss: {avg_train_loss:.5f} | Valid Loss: {valid_loss:.5f}')
        
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = model.state_dict()
            early_counter = 0
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                print(f"[UDA] Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # --- Phase 2: Few-Shot Fine-Tuning ---
    print("\n--- Phase 2: Target Domain Fine-Tuning ---")
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    ft_criterion = nn.MSELoss()
    
    for epoch in range(1, args.ft_epoch + 1):
        model.train()
        total_ft_loss = 0
        for data, label in ft_loader:
            data, label = data.to(device).float(), label.to(device).float()
            xc, xd = data[:, :4, :], data[:, 4:, :]
            
            soh_pred, _ = model(xc, xd, alpha=0.0)
            loss = ft_criterion(soh_pred, label)
            
            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
            total_ft_loss += loss.item() * data.size(0)
            
        if epoch % 10 == 0:
            avg_ft_loss = total_ft_loss / len(ft_loader.dataset)
            print(f"FT Epoch [{epoch:03d}/{args.ft_epoch}] Loss: {avg_ft_loss:.6f}")

    # --- Phase 3: Evaluation ---
    print("\n--- Phase 3: Final Evaluation ---")
    _, y_true, y_pred = evaluate_soh(model, tgt_eval_loader, device)
    
    MAE, MAPE, MSE, R2 = eval_metrics(y_true, y_pred)
    print(f"[Results] Target Batch {args.test_batch} Overall: MAE={MAE:.5f}, MAPE={MAPE:.5f}, MSE={MSE:.5f}, R2={R2:.5f}\n")

    unique_bids = np.unique(eval_batt_ids)
    per_batt_metrics = []
    all_y_true_plot, all_y_pred_plot = [], []

    for bid in unique_bids:
        mask = (eval_batt_ids == bid)
        y_t, y_p = y_true[mask], y_pred[mask]

        all_y_true_plot.append(y_t)
        all_y_pred_plot.append(y_p)

        MAE_b, MAPE_b, MSE_b, R2_b = eval_metrics(y_t, y_p)
        per_batt_metrics.append([bid, MAE_b, MAPE_b, MSE_b, R2_b])
        print(f"  Battery {int(bid)}: MAE={MAE_b:.5f}, MAPE={MAPE_b:.5f}, R2={R2_b:.5f}")

    # Save results
    save_dir = os.path.join(args.save_folder, f'train{"-".join(map(str, args.train_batches))}_test{args.test_batch}')
    os.makedirs(save_dir, exist_ok=True)

    img_save_path = os.path.join(save_dir, 'soh_predictions_curve.png')
    plot_8_batteries(all_y_true_plot, all_y_pred_plot, save_path=img_save_path)
    
    np.savez(
        os.path.join(save_dir, 'results.npz'),
        true_label=y_true,
        pred_label=y_pred,
        test_errors=np.array([MAE, MAPE, MSE, R2]),
        per_batt_metrics=np.array(per_batt_metrics, dtype=np.float32),
    )
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_ssda_model.pth'))
    print(f"\n[Info] All results saved to: {save_dir}")

if __name__ == '__main__':
    main()