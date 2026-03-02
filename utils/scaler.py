import numpy as np

class Scaler:
    def __init__(self , data):

        self.data = data

        # 根据数组维度选择处理方式
        if self.data.ndim == 3:             # (N , C , L)
            self.mean = self.data.mean(axis = (0 , 2)).reshape(1 , -1 ,1)
            self.var = self.data.var(axis = (0 , 2)).reshape(1 , -1 ,1)
            self.max = self.data.max(axis = (0 , 2)).reshape(1 , -1 ,1)
            self.min = self.data.min(axis = (0 , 2)).reshape(1 , -1 ,1)

        elif self.data.ndim == 2:           # (N , C)
            self.mean = self.data.mean(axis=0).reshape(1, -1)
            self.var  = self.data.var(axis=0).reshape(1, -1)
            self.max  = self.data.max(axis=0).reshape(1, -1)
            self.min  = self.data.min(axis=0).reshape(1, -1)
        
        else:
            raise ValueError('data dim error ! (只支持 (N , C , L)或者 (N , C) 两种形状)')

    def standerd(self):
        """
        标准化： (x - mean) / (var + eps)
        """
        X = (self.data - self.mean) / (self.var + 1e-6)
        return X

    def minmax(self , feature_range = (0 , 1)):
        """
        Min-Max归一化：
        范围是(0,1)缩放到[0,1]
        X_norm = (X - min) / (max - min)
        范围是(-1,1)缩放到[-1,1]
        X_norm = 2 * (X - min) / (max - min) - 1
        """
        if feature_range == (0 , 1):
            X = (self.data - self.min)/((self.max - self.min) + 1e-6)
        elif feature_range == (-1 , 1):
            X = 2 * (self.data - self.min)/((self.max - self.min) + 1e-6) - 1
        else:
            raise ValueError('feature_range error! 只能是 (0,1) 或 (-1,1)')
        return X
        


        
