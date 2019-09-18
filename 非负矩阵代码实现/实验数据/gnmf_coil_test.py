 # coding: utf-8
import sys
import codecs
sys.stdout=codecs.getwriter('utf8')(sys.stdout.detach())


import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import warnings
import matplotlib
from PIL import Image
from pylab import *
from numpy import *
from numpy import *
import os
from scipy.io import loadmat
import pandas as pd
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import  normalize
from  sklearn.preprocessing import  Normalizer
from matplotlib import pyplot as plt
import networkx as nx
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# import split_list
import scipy.io as scio
# from sklearn.decomposition import NMF




 
def matrix_factorisation(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
	Q = Q.T
	for step in range(steps):
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - np.dot(P[i,:],Q[:,j])
					for k in range(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P,Q)
		e = 0
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
					for k in range(K):
						e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break
	return P, Q.T


def read_data():
	# 读取图像
	# def get_imlist(path):   #此函数读取特定文件夹下的png格式图像
	#     return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

	# c=get_imlist(r"./coil-20-proc")    #r""是防止字符串转译
	# print(c)     #这里以list形式输出bmp格式的所有图像（带路径）
	# d=len(c)    #这可以以输出图像个数


	# data=np.empty((d,128*128)) #建立d*（28*28）的矩阵
	# while d>0:
	#     img=Image.open(c[d-1])  #打开图像
	#     #img_ndarray=numpy.asarray(img)
	#     img_ndarray=np.asarray(img,dtype='float64') #将图像转化为数组并将像素转化到0-1之间
	#     print(img_ndarray)
	#     # data[d-1]=np.ndarray.flatten(img_ndarray)    #将图像的矩阵形式转化为一维数组保存到data中
	#     # data[d-1] = img_ndarray
	#     d=d-1
	# print(data)

	# # A=np.array(data[0]).reshape(28,28)   #将一维数组转化为矩28*28矩阵
	# #print A
	# savetxt('num7.txt',A,fmt="%.0f") #将矩阵保存到txt文件中

	# # 直接读取mat
	# m = loadmat(r"./coil20.mat")
	# print(m.keys())
	# df = pd.DataFrame(m["fea"])
	# print(df.head())

	# trueClass = np.zeros((1440,1))
	# def get_imlist(path):   #此函数读取特定文件夹下的png格式图像，返回图片所在路径的列表
	# 	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
	# c = get_imlist(r"./coil-20-proc")    #r""是防止字符串转译
	# print(c[0])
	# print(c)     #这里以list形式输出jpg格式的所有图像（带路径）
	# d = len(c)    #输出图像个数，共有1440张图片，每张是128*128像素，共有20个类
	# for i in range(d):
	# 	newC = c[i].split("j")[1].split("_")[0]
	# 	# trueClass.append(newC)
	# 	trueClass[i][0] = newC
	# trueClass = np.array(trueClass).T[0]
	# print(trueClass)
	# print("The picture number is",d)
	#
	#
	# data = np.zeros((d , 128*128)) #建立d*（128,128）的矩阵
	# for i in range(d):
	# 	# print(c[i])
	# 	img = Image.open(c[i])  #打开图像
	# 	# img = img2vector(c[i - 1])
	# 	#img_ndarray = numpy.asarray(img)
	# 	img_ndarray = np.asarray(img,dtype='float64')/255  #将图像转化为数组并将像素转化到0-1之间
	# 	# print("asd",img_ndarray)
	# 	data[i] = np.ndarray.flatten(img_ndarray)    #将图像的矩阵形式保存到data中
	# 	# print("data",d,"is",data[d-1])
	# 	# data[:,i] = img[0]
	# data = data.T
	# print("data.shape:",data.shape)

	path = 'label.mat'
	path1 = 'fea.mat'
	dataA = scio.loadmat(path)
	dataB = scio.loadmat(path1)
	# print(dataA.keys())
	# print(dataB.keys())
	# print("asdAsdasd",dataB['fea'])
	# print(dataA['label'])
	data = dataB['fea']
	trueClass = dataA['label']

	return data,trueClass
	# print(data.type())



def load_data(file_path):
	f = open(file_path)
	V = []
	for line in f.readlines():
		lines = line.strip().split("\t")
		data = []
		for x in lines:
			data.append(float(x))
		V.append(data)
	return mat(V)


def img2vector(imgFile):
	# img = Image.open(imgFile).convert('L')
	# img_arr = np.array(img, 'i') # 128px * 128px 灰度图像
	# img_normlization = np.round(img_arr/255) # 对灰度值进行归一化
	# img_arr2 = np.reshape(img_normlization, (-1,1)) # 1 * 16384 矩阵
	# print(img_arr2.shape)
	# return img_arr2
	returnVect = zeros((1, 16384))
	fr = open(imgFile,'rb')
	for i in range(128):
		lineStr = fr.readline()
		for j in range(128):
			returnVect[0, 128 * i + j] = int(lineStr[j])
	print(returnVect)
	return returnVect


def myKNN(S, k, sigma=1.0):
    N = len(S)#输出的是矩阵的行数
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0], reverse = True)
        # print(dist_with_index)
        neighbours_id = [dist_with_index[m][1] for m in range(k)] # xi's k nearest neighbours
        # print("neigh",neighbours_id)
        for j in neighbours_id: # xj is xi's neighbour
            # print(j)
            A[i][j] = 1
            A[j][i] = A[i][j] # mutually
        # print(A[i])
    m = np.shape(A)[0]
    print(m)
    for i in range(m):
        for j in range(m):
            if j == i:
                A[i][j] = 0

    return A

#计算欧几里得距离（是计算相似度的一种方法，其他方法还有：曼哈顿距离、皮尔逊相关度、Jaccard系数、Tanimoto系数等）
def euclidean(p,q):
	#如果两数据集数目不同，计算两者之间都对应有的数
	same = 0
	for i in p:
		if i in q:
			same += 1

	#计算欧几里德距离,并将其标准化
	e = sum([(p[i] - q[i]) ** 2 for i in range(same)])
	return 1/(1+e**.5)#标准化


#训练相似矩阵W
def trainW(v):
	similarMatrix = cosine_similarity(v.T)
	# similarMatrix = pairwise_distances(v,metric="cosine")
	m = np.shape(similarMatrix)[0]
	print(m)
	for i in range(m):
		for j in range(m):
			if j == i:
				similarMatrix[i][j] = 0
	return similarMatrix




def train(V, r, k):
	m, n = shape(V)
	#先随机给定一个W、H，保证矩阵的大小
	W = np.array(random.random((m, r)))
	H = np.array(random.random((n, r)))

	# minMax = MinMaxScaler()
	# W = minMax.fit_transform(W)

	# 根据p紧邻图制作权重矩阵
	# neigh = NearestNeighbors(n_neighbors = 5)
	# neigh.fit(V)
	# A = kneighbors_graph(V.T, n_neighbors = 5, mode = 'distance', metric = 'minkowski', p = 2)
	# A = A.toarray()
	# print("A.shape:", A.shape)
	# HangA = A.shape[0]
	# LieA = A.shape[1]
	# kneighbors_graph([X,n_neighbors,mode]) 计算X中k个临近点（列表）对应的权重。
	# metric：字符或者调用，默认值为‘minkowski’
	# n_neighbors：整数，可选（默认值为5）,用kneighbors_graph查找的近邻数。
	# p：整数，可选（默认值为2）。是sklearn.metrics.pairwise.pairwise_distance里的闵可夫斯基度量参数，当 p=1时，
	# 使用曼哈顿距离。当p=2时，使用的是欧氏距离。对于任意的p，使用闵可夫斯基距离。

	D = []
	trainV = V
	similarMatrix = trainW(trainV)
	print("similarM.shape:",similarMatrix.shape)
	linMatrix = myKNN(similarMatrix,5)

	print("最近邻矩阵：",linMatrix)
	print("最近邻矩阵的规格：",linMatrix.shape)
	# AV = linMatrix.T
	# G = nx.Graph()
	# for i in range(len(linMatrix)):
	# 	for j in range(len(linMatrix)):
	# 		G.add_edge(i,j)
	# AW = pairwise_distances(linMatrix, metric="euclidean")
	# vectorizer = np.vectorize(lambda x: 1 if x < 5 else 0)
	# quanMatrix = np.vectorize(vectorizer)(AW)
	# AW = kneighbors_graph(V, n_neighbors=5, mode='distance', metric='minkowski', p=2, include_self=True)
	# print("加上权重之后的邻接矩阵",quanMatrix)

	# degree matrix
	D = np.diag(np.sum(linMatrix, axis=0))
	# D = np.diag(np.sum(np.array(AW.todense()), axis=1))
	print('degree matrix:',D)
	print("度矩阵的规格：",D.shape)

	#K为迭代次数
	for x in range(k):

	#权值更新
		# a = V.T * W + 100 * linMatrix * H
		# b = H * W.T * W + 100 * D * H
		# print("a/b",a/b)
		# print("H",H)
		# lsasd = (a/b).reshape(20,1440)
		# Hnew = np.dot(H , lsasd)
		#
		# c = V * H
		# d = W * H.T * H
		# lasd = (c/d).reshape(20,16384)
		# Wnew = np.dot(W , lasd)
		# a = np.multiply(V.T, W) + 100 * np.multiply(linMatrix, H)
		# b = np.multiply(np.multiply(H, W.T), W) + 100 * np.multiply(D, H)
		# H = np.multiply(H , asd)
		a = np.dot(V.T, W) + 100 * np.dot(linMatrix, H)
		b = np.dot(np.dot(H, W.T), W) + 100 * np.dot(D, H)
		# H = np.multiply(H , a/b)
		# W = np.multiply(W , c/d)
		for i_1 in range(n):
			for j_1 in range(r):
				if b[i_1, j_1] != 0:
					H[i_1, j_1] = H[i_1, j_1] * (a[i_1, j_1] / b[i_1, j_1])
		# c = V * H
		# d = W * H.T * H
		c = np.dot(V, H)
		d = np.dot(np.dot(W, H.T), H)
		for i_2 in range(m):
			for j_2 in range(r):
				if d[i_2, j_2] != 0:
					W[i_2, j_2] = W[i_2, j_2] * (c[i_2, j_2] / d[i_2, j_2])
	return W,H


def NMI(A,B):
	#样本点数
	total = len(A)
	A_ids = set(A)
	B_ids = set(B)
	#互信息计算
	MI = 0
	eps = 1.4e-45
	for idA in A_ids:
		for idB in B_ids:
			idAOccur = np.where(A==idA)
			idBOccur = np.where(B==idB)
			idABOccur = np.intersect1d(idAOccur,idBOccur)
			px = 1.0*len(idAOccur[0])/total
			py = 1.0*len(idBOccur[0])/total
			pxy = 1.0*len(idABOccur)/total
			MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
	# 标准化互信息
	Hx = 0
	for idA in A_ids:
		idAOccurCount = 1.0*len(np.where(A==idA)[0])
		Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
	Hy = 0
	for idB in B_ids:
		idBOccurCount = 1.0*len(np.where(B==idB)[0])
		Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
	MIhat = 2.0*MI/(Hx+Hy)
	return MIhat

def NMF(X, r, lamb=0.2, maxit=500):
    if not((r < X.shape[0]) | (r < X.shape[1])):
        raise ValueError('Erro: Valor de r')
    H = np.random.rand(r, X.shape[1])
    D = np.random.rand(X. shape[0], r)
    Dnorm = D / np.sum(D**2, axis=0)**(.5)
    for i in range(maxit):
        H = H * (np.dot(Dnorm.T, X)) / (np.dot(np.dot(Dnorm.T, Dnorm), H) + lamb)
        D = Dnorm * (np.dot(X, H.T) + Dnorm * (np.dot(np.ones((X.shape[0], X.shape[0])), np.dot(Dnorm, np.dot(H, H.T)) * Dnorm))) / (np.dot(Dnorm, np.dot(H, H.T)) + Dnorm * (np.dot(np.ones((X.shape[0], X.shape[0])), np.dot(X, H.T) * Dnorm)))
        Dnorm = D / np.sum(D**2, axis=0)**(.5)
    return D, H


if __name__ == "__main__":
	R ,trueClass= read_data()
	trueClass = np.array(trueClass.T)[0]
	print("R.shape",R.shape)
	print("trueClass.shape",trueClass.shape)
	# print(R)

	# print(R)
	# N = len(R)
	# M = len(R[0])
	# K = 2
	# P = np.random.rand(N,K)
	# Q = np.random.rand(M,K)
	# nP, nQ = matrix_factorisation(R, P, Q, K)
	# nR = np.dot(np,nQ.T)
	# print(nR)

	# R_final = normalize(R, norm='l1')
	# transformer = Normalizer().fit(R)
	# R_final = transformer.transform(R)

	minMax = MinMaxScaler()
	R_final = minMax.fit_transform(R)
	W, H = train(R_final.T, 20, 100)


	#对算法得到的H矩阵归一化
	AnominMax = MinMaxScaler()
	H_final = AnominMax.fit_transform(H)
	print("R的规格：",R.shape)
	print("W的规格：",W.shape)
	print("H的规格：",H.shape)
	print(H)
	model_kmeans = KMeans(n_clusters=20)  #建立模型对象
	#训练聚类模型
	y_pre = model_kmeans.fit(H_final).labels_   #预测聚类模型
	print("trueClass:",trueClass)
	print("y_pre:",y_pre)
	print(y_pre.shape)
	# for i in range(len(y_pre)):
	# 	y_pre[i] = y_pre[i] + 1
	# 	print(y_pre)

	# R_pred = split_list.splitlist(np.array(R_new.ravel()))//R_new.ravel()出来的结果是一个嵌套的一维数组，按传统的分割数组的方法可以，但是1440*16384的规格太大了，电脑不行。。
	# R_pred = np.array(R_new.ravel())[0]
	# R_true = np.array(R.flatten())
	
	# print(R_true)
	# print(R_pred)
	# print(isinstance(R_pred,list))
	result_NMI = metrics.normalized_mutual_info_score(trueClass, y_pre)
	print("NMI:",result_NMI)



	'''
		NMF
	'''

	nmfW, nmfH = NMF(R_final.T, 20,lamb=0,maxit=100)
	#对nmf算法得到的nmfH进行归一化
	nmfminMax = MinMaxScaler()
	nmfH_final = nmfminMax.fit_transform(nmfH)

	nmf_pre = model_kmeans.fit(nmfH_final.T).labels_
	nmfResult_NMI = metrics.normalized_mutual_info_score(trueClass, nmf_pre)
	print("NMF's NMI:",nmfResult_NMI)

