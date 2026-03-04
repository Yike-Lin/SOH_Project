import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from dataloader.xjtu_loader import XJTUDataset
from models.lstm import SOHLSTM
from utils.metrics import AverageMeter, eval_metrics



"""
从命令行获取超参数和配置
"""
def get_args():
    parser = argparse.ArgumentParser(description='SOH estimation with LSTM on XJTU')

    # 数据设置
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU'])
    parser.add_argument('--input_type', type=str, default='charge',
                        choices=['charge', 'partial_charge', 'handcraft_features'])
    parser.add_argument('--test_battery_id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--normalized_type', type=str, default='minmax',
                        choices=['minmax', 'standard'])
    parser.add_argument('--minmax_range', type=tuple, default=(-1, 1),
                        choices=[(0, 1), (-1, 1)])

    parser.add_argument('--batch', type=int, default=1,choices=[1,2,3,4,5,6,7,8,9])

    # 模型&训练设置
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_folder', default='results_LSTM')
    parser.add_argument('--experiment_num', type=int, default=1,help='同一配置重复实验次数')

    args = parser.parse_args()
    return args


"""
根据 args 加载数据
"""
def load_data(args):
    dataset = XJTUDataset(args)

    if args.input_type == 'charge':
        loaders = dataset.get_charge_data(test_battery_id = args.test_battery_id)
    elif args.input_type == 'partial_charge':
        loaders = dataset.get_partial_data(test_battery_id = args.test_battery_id)
    else:
        loaders = dataset.get_features(test_battery_id = args.test_battery_id)

    return loaders


"""
训练模型一个epoch
"""
def train_one_epoch(model , train_loader , optimizer , criterion , device):
    # 设置
    model.train()
    meter = AverageMeter()

    # 批量拿数据
    for data , label in train_loader:
        data = data.to(device).float()          # (B, 4, L)
        label = label.to(device).float()        # (B, 1)

        # 前向传播
        pred = model(data)
        loss = criterion(pred , label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录与返回
        meter.update(loss.item() , n = data.size(0))
    
    return meter.avg


"""
评估函数
"""
def evaluate(model , loader , criterion , device):
    model.eval()
    meter = AverageMeter()

    true_list = []
    pred_list = []

    with torch.no_grad():
        for data , label in loader:
            data = data.to(device).float()
            label = label.to(device).float()

            pred = model(data)
            loss = criterion(pred , label)

            meter.update(loss.item() , n = data.size(0))

            true_list.append(label.cpu().numpy())
            pred_list.append(pred.cpu().numpy())

    true_arr = np.concatenate(true_list , axis=0)
    pred_arr = np.concatenate(pred_list , axis=0)

    return meter.avg , true_arr , pred_arr



"""
main()函数
"""
def main():
    # 1. 读参数
    args = get_args()

    # 2. 创建保存结果的目录
    os.makedirs(args.save_folder , exist_ok=True)

    # 3. 固定随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # 4. 选择设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device:' , device)

    # 允许重复实验多次(experiment_num)
    for exp in range(args.experiment_num):
        print(f'\n=== Experiment {exp + 1}/{args.experiment_num} ===')

        # 5. 加载数据
        data_loader = load_data(args)
        train_loader = data_loader['train']
        valid_loader = data_loader['valid']
        test_loader = data_loader['test']

        # 6. 构建LSTM模型
        model = SOHLSTM(input_channels = 4 , seq_len = 128,
                        hidden_size = 128 , num_layers = 2).to(device)

        # 7. 定义损失函数，优化器，学习率调度器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = args.lr,
            weight_decay = args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = [30 , 70],
            gamma = 0.5
        )

        best_valid_loss = float('inf')  # 当前看到的最小验证集 loss
        best_state_dict = None          # 保存最佳模型参数
        best_test_result = None         # 保存最佳模型在测试集上的结果

        early_stop_counter = 0          # 早停计数器
        train_losses, valid_losses = [], []

        # 8. 训练循环
        for epoch in range(1 , args.n_epoch + 1):
            # 训练一轮
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            # 在验证集上评估
            valid_loss, _, _ = evaluate(model, valid_loader, criterion, device)

            # 学习率调度
            scheduler.step()

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f'Epoch [{epoch}/{args.n_epoch}] '
                  f'train_loss={train_loss:.5f} valid_loss={valid_loss:.5f}')

            # 早停 & 记录最优模型
            if valid_loss < best_valid_loss:
                # 找到新的更优验证集结果
                best_valid_loss = valid_loss
                early_stop_counter = 0

                # 用当前模型在测试集上评估一次
                test_loss, true_test, pred_test = evaluate(model, test_loader, criterion, device)
                print(f'  -> new best valid, test_loss={test_loss:.5f}')

                # 保存这一次的模型参数和测试结果
                best_state_dict = model.state_dict()
                best_test_result = (true_test, pred_test, test_loss)
            else:
                # 验证集没变好，早停计数+1
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop:
                    print('Early stopping triggered.')
                    break
        
        # 9. 训练结束，使用在验证集上表现最好的那次模型，统计测试指标
        true_test, pred_test, test_loss = best_test_result
        MAE, MAPE, MSE, R2 = eval_metrics(true_test, pred_test)
        print(f'Final test: MAE={MAE:.5f} , MAPE={MAPE:.5f} , MSE={MSE:.5f}, R2={R2:.5f}')

        # 10. 保存结果和模型
        save_dir = os.path.join(
            args.save_folder,
            f'{args.data}-{args.input_type}',
            f'LSTM-batch{args.batch}-test{args.test_battery_id}-exp{exp + 1}'
        )
        os.makedirs(save_dir , exist_ok = True)

        # 保存训练过程的 loss 以及测试集的预测结果
        np.savez(
            os.path.join(save_dir, 'results.npz'),
            train_loss=np.array(train_losses),
            valid_loss=np.array(valid_losses),
            true_label=true_test,
            pred_label=pred_test,
            test_errors=np.array([MAE, MAPE, MSE, R2])
        )
        # 保存最佳模型参数
        torch.save(best_state_dict, os.path.join(save_dir, 'best_model.pth'))
        
if __name__ == '__main__':
    main()
