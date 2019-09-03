# 准确率
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3,9,9,8,5,8]
y_true = [0, 1, 2, 3,2,6,3,5,9]

print(accuracy_score(y_true, y_pred))


print(accuracy_score(y_true, y_pred, normalize=False))  # 类似海明距离，每个类别求准确后，再求微平均

