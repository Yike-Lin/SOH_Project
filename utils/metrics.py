import numpy as np
from sklearn import metrics

class AverageMeter:

    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        重置内部统计量,在开始新的epoch时调用
        """
        self.val = 0        # 当前值
        self.avg = 0        # 平均值
        self.sum = 0        # 累加和
        self.count = 0      # 样本数量总和
    
    def update(self , val , n = 1):
        """
        动态计算加权平均值
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.num / self.count
    
    def eval_metrics(true_label , pred_label):
        """
        计算评估指标
        - MAE  : 平均绝对误差
        - MAPE : 平均绝对百分比误差
        - MSE  : 均方误差
        - R2 : 决定系数
        """
        MAE = metrics.mean_absolute_error(true_label , pred_label)
        MAPE = metrics.mean_absolute_percentage_error(true_label , pred_label)
        MSE = metrics.mean_squared_error(true_label , pred_label)
        R2 = metrics.r2_score(true_label, pred_label)
        return MAE , MAPE , MSE , R2


