# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/9/23 11:05
# @File     : nmf_pets.py.py
# @Software : PyCharm
import random

import numpy as np
import matplotlib.pyplot as pl
import nmf
import nimfa
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.preprocessing import MinMaxScaler


# 读取txt文件
def ReadTxtData(filePath):
    resultData = []
    with open(filePath,"r") as f:
        for line in f:
            resultData.append(list(line.strip("\n").split("::")))
    # print(len(resultData))
    print(resultData)
    return resultData

def main(filePath):
    dataSet = ReadTxtData(filePath)
    user_item_matrix = np.zeros((1624, 1672))
    for i in range(len(dataSet)):
        m = int(dataSet[i][0])
        n = int(dataSet[i][1])
        r = int(dataSet[i][2])
        user_item_matrix[m][n] = r
    print("dddd", user_item_matrix)
    user_item_matrix = user_item_matrix
    hang, lie = np.shape(user_item_matrix)
    train_matrix = user_item_matrix
    test_matrix = np.zeros((1624, 1672))
    line = np.argwhere(train_matrix > 0)  # 3093*2
    # print(line.shape)
    # for i in range(hang):
    # 	asd = []
    # 	asd.append(np.argwhere(train_matrix[i] > 0))
    # 	# print("ksdj", asd[0])
    # 	if len(asd[0]) == 1:
    # 		test_matrix[i][asd[0][0]] = train_matrix[i][asd[0][0]]
    # 		train_matrix[i][asd[0][0]] = 0
    # 	elif len(asd[0]) > 1 and len(asd[0]) <= 3:
    # 		rnd = random.sample(range(1, len(asd[0])), 1)
    # 		test_matrix[i][asd[0][rnd[0]]] = train_matrix[i][asd[0][rnd[0]]]
    # 		train_matrix[i][asd[0][rnd[0]]] = 0
    # 	else:
    # 		rnd = random.sample(range(1, len(asd[0])), 2)
    # 		# print("rnd", rnd[0])
    # 		# print("adasdsd", asd[0][rnd[0]])
    # 		test_matrix[i][asd[0][rnd[0]]] = train_matrix[i][asd[0][rnd[0]]]
    # 		train_matrix[i][asd[0][rnd[0]]] = 0
    # 		test_matrix[i][asd[0][rnd[1]]] = train_matrix[i][asd[0][rnd[1]]]
    # 		train_matrix[i][asd[0][rnd[1]]] = 0
    for i in range(hang):
        randomResult = random.sample(range(1, 1672), 200)
        for j in range(len(randomResult)):
            o = randomResult[j]
            test_matrix[i][o] = train_matrix[i][o]
            train_matrix[i][o] = 0
    trc = np.array(np.argwhere(train_matrix > 0))
    print("trc:\n", trc)
    lentrc = len(trc)
    print("trc.length:\n", lentrc)
    # print("each:\n",trc[2][1])
    total = 0
    for i in range(lentrc):
        total += train_matrix[trc[i][0]][trc[i][1]]
    # print(total)
    trainR = total / lentrc
    print("trainR:\n", trainR)

    nmf = nimfa.Nmf(train_matrix, max_iter=20, rank=10, update='euclidean', objective='fro')
    nmf_fit = nmf()
    W = nmf_fit.basis()
    H = nmf_fit.coef()

    nnewX = np.array(np.dot(W, H))
    nnewX = [[nnewX[i][j] + trainR for j in range(len(nnewX[i]))] for i in range(len(nnewX))]  # 每个元素累加r
    # newX = AnominMax.fit_transform(newX)
    xiabao = np.argwhere(test_matrix > 0)  # 获取测试集中值大于0的元素的下标
    print(xiabao)
    y_true = []
    y_pred = []
    for i, j in xiabao:
        y_true.append(test_matrix[i][j])
        y_pred.append(nnewX[i][j])
    print("y_pred", y_pred)
    print("y_true", y_true)
    print("RMSE:", mean_squared_error(y_true, y_pred) ** (.5))
    print("MAE:", mean_absolute_error(y_true, y_pred))

if __name__ == "__main__":
    filePath = "./Yelp/pets/ratings.txt"
    main(filePath)