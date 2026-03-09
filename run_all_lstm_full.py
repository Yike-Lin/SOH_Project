import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from dataloader.xjtu_loader import XJTUDataset
from utils.metrics import AverageMeter, eval_metrics

from models.Multi_Bi_LSTM_Attention import DualStreamMultiBiLSTMAttention

def get_args():
    parser = argparse.ArgumentParser(
        description='Run Dual-Stream SOH estimation (full charge+discharge) for all batches/tests/experiments'
    )

    # 数据设置
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU'])

    parser.add_argument('--input_type', type=str, default='full',
                        choices=['full', 'charge', 'partial_charge', 'handcraft_features'],
                        help='本脚本主要用于 full，如果想跑别的类型可以改这个')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--normalized_type', type=str, default='minmax',
                        choices=['minmax', 'standard'])
    parser.add_argument('--minmax_range', type=tuple, default=(-1, 1),
                        choices=[(0, 1), (-1, 1)])
    parser.add_argument('--batch_list', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='要跑的 batch 列表')
    parser.add_argument('--test_ids', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='每个 batch 要跑的 test_battery_id 列表')

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--save_folder', default='results_DualStream_Auto') 
    
    parser.add_argument('--experiment_num', type=int, default=3,
                        help='同一配置重复实验次数')

    args = parser.parse_args()
    return args


def load_data(args):
    dataset = XJTUDataset(args)

    if args.input_type == 'charge':
        loaders = dataset.get_charge_data(test_battery_id=args.test_battery_id)
    elif args.input_type == 'partial_charge':
        loaders = dataset.get_partial_data(test_battery_id=args.test_battery_id)
    elif args.input_type == 'full':
        loaders = dataset.get_full_data(test_battery_id=args.test_battery_id)
    else:
        loaders = dataset.get_features(test_battery_id=args.test_battery_id)

    return loaders


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    meter = AverageMeter()

    for data, label in train_loader:
        data = data.to(device).float()     # (B, 8, L)
        label = label.to(device).float()   # (B, 1)

        # 数据切片
        x_charge = data[:, 0:4, :]
        x_discharge = data[:, 4:8, :]

        # 传入两个数据流
        pred = model(x_charge, x_discharge)

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), n=data.size(0))

    return meter.avg


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()

    true_list = []
    pred_list = []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device).float()
            label = label.to(device).float()

            x_charge = data[:, 0:4, :]
            x_discharge = data[:, 4:8, :]

            # 传入两个数据流
            pred = model(x_charge, x_discharge)

            loss = criterion(pred, label)

            meter.update(loss.item(), n=data.size(0))

            true_list.append(label.cpu().numpy())
            pred_list.append(pred.cpu().numpy())

    true_arr = np.concatenate(true_list, axis=0)
    pred_arr = np.concatenate(pred_list, axis=0)

    return meter.avg, true_arr, pred_arr


def main():
    args = get_args()
    
    if args.input_type != 'full':
        raise ValueError("使用双流模型时，请确保 --input_type 设置为 'full'！")

    os.makedirs(args.save_folder, exist_ok=True)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Batches:', args.batch_list)
    print('Test IDs per batch:', args.test_ids)
    print('Experiments per config:', args.experiment_num)
    print('Input type:', args.input_type)

    # 遍历所有 batch
    for batch_id in args.batch_list:
        # 更新 args.batch
        args.batch = batch_id
        print(f'\n===== Batch {batch_id} =====')

        # 遍历所有测试电池
        for test_id in args.test_ids:
            args.test_battery_id = test_id
            print(f'\n--- Batch {batch_id}, Test battery {test_id} ---')

            # 为了避免某些 batch/test_id 组合不存在，先尝试加载一次数据
            try:
                data_loader = load_data(args)
            except Exception as e:
                print(f'  跳过: batch={batch_id}, test_id={test_id}, 加载数据失败: {e}')
                continue

            train_loader = data_loader['train']
            valid_loader = data_loader['valid']
            test_loader = data_loader['test']

            # 多次实验
            for exp in range(args.experiment_num):
                print(f'\n>>> Batch {batch_id} | Test {test_id} | Experiment {exp + 1}/{args.experiment_num}')

                # 针对每一次实验重设随机种子（可选）
                seed = args.random_seed + exp
                torch.manual_seed(seed)
                np.random.seed(seed)

                x0, _ = next(iter(train_loader))
                if x0.dim() != 3 or x0.size(1) != 8:
                    raise ValueError(f'双流模型期望输入 8 通道数据 (4充+4放)，但得到 {x0.shape}')
                
                L = x0.size(2)
                print(f'  Inferred input shape: Total Channels=8, L={L}')

                model = DualStreamMultiBiLSTMAttention(
                    input_channels=4, 
                    seq_len=L,
                    hidden_size=128, 
                    num_layers=2
                ).to(device)

                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[30, 70],
                    gamma=0.5
                )

                best_valid_loss = float('inf')
                best_state_dict = None
                best_test_result = None
                early_stop_counter = 0
                train_losses, valid_losses = [], []

                # 训练循环
                for epoch in range(1, args.n_epoch + 1):
                    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                    valid_loss, _, _ = evaluate(model, valid_loader, criterion, device)
                    scheduler.step()

                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)

                    print(f'  Epoch [{epoch}/{args.n_epoch}] '
                          f'train_loss={train_loss:.5f} valid_loss={valid_loss:.5f}')

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        early_stop_counter = 0

                        test_loss, true_test, pred_test = evaluate(model, test_loader, criterion, device)
                        print(f'    -> new best valid, test_loss={test_loss:.5f}')

                        best_state_dict = model.state_dict()
                        best_test_result = (true_test, pred_test, test_loss)
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= args.early_stop:
                            print('    Early stopping triggered.')
                            break

                if best_test_result is None:
                    print('  Warning: best_test_result is None, 跳过保存。')
                    continue

                true_test, pred_test, test_loss = best_test_result
                MAE, MAPE, MSE, R2 = eval_metrics(true_test, pred_test)
                print(f'  Final test: MAE={MAE:.5f} , MAPE={MAPE:.5f} , '
                      f'MSE={MSE:.5f}, R2={R2:.5f}')

                save_dir = os.path.join(
                    args.save_folder,
                    f'{args.data}-full',
                    f'batch{batch_id}',
                    f'test{test_id}',
                    f'exp{exp + 1}'
                )
                os.makedirs(save_dir, exist_ok=True)

                np.savez(
                    os.path.join(save_dir, 'results.npz'),
                    train_loss=np.array(train_losses),
                    valid_loss=np.array(valid_losses),
                    true_label=true_test,
                    pred_label=pred_test,
                    test_errors=np.array([MAE, MAPE, MSE, R2])
                )
                torch.save(best_state_dict, os.path.join(save_dir, 'best_model.pth'))
                print(f'  Saved results to: {save_dir}')


if __name__ == '__main__':
    main()