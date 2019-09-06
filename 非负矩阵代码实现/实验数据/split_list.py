# -*- coding:utf-8 -*-
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import scipy.io as scio

path = 'COIL20.mat'
data = scio.loadmat(path)
print(data.keys())
print(data['fea'])
print(data['gnd'])
print(type(data['fea']))


# class BinJianAna:
#     def __init__(self):
#         pass
#
#     def ComuTF(self, lst1, lst2):
#         # 计算TPR和FPR
#         # lst1为真实值,lst2为预测值
#         TP = sum([1 if a == b == 1 else 0 for a, b in zip(lst1, lst2)])  # 正例被预测为正例
#         FN = sum([1 if a == 1 and b == 0 else 0 for a, b in zip(lst1, lst2)])  # 正例被预测为反例
#         TPR = TP / (TP + FN)
#         TN = sum([1 if a == b == 0 else 0 for a, b in zip(lst1, lst2)])  # 反例被预测为反例
#         FP = sum([1 if a == 0 and b == 1 else 0 for a, b in zip(lst1, lst2)])  # 反例被预测为正例
#         FPR = FP / (TN + FP)
#         return TPR - FPR
#
#     def Getps_ks(self, real_data, data):
#         # real_data为真实值，data为原数据
#         d = []
#         for i in data:
#             pre_data = [1 if line >= i else 0 for line in data]
#             d.append(self.ComuTF(real_data, pre_data))
#         return max(d), data[d.index(max(d))]
#
#
# if __name__ == '__main__':
#     a = BinJianAna()
#     data = [790, 22, 345, 543, 564, 342, 344, 666, 789, 123, 231, 234, 235, 347, 234, 237, 178, 198, 567, 222]  # 原始评分数据
#     real_data = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
#     y_pred_prob = [0.42, 0.73, 0.55, 0.37, 0.57, 0.70, 0.25, 0.23, 0.46, 0.62, 0.76, 0.46, 0.55, 0.56, 0.56, 0.38, 0.37,
#                    0.73, 0.77, 0.21]
#     print(len(data))
#     print(a.Getps_ks(real_data, data))  # 自己实现

# data = h5py.File("./COIL20.mat")