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
# import gnmf
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
# import seaborn as sns


import split_list


 
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


	def get_imlist(path):   #此函数读取特定文件夹下的jpg格式图像，返回图片所在路径的列表
		return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
	c=get_imlist(r"./coil-20-proc")    #r""是防止字符串转译
	print (c)     #这里以list形式输出jpg格式的所有图像（带路径）
	d=len(c)    #这可以以输出图像个数，如果你的文件夹下有698张图片，那么d为698
	print("The picture number is",d)
	
	 
	data=np.empty((d,128*128)) #建立d*（128,128）的矩阵
	print(data)
	while d>0:
		img=Image.open(c[d-1])  #打开图像
		
		#img_ndarray=numpy.asarray(img)
		img_ndarray=np.asarray(img,dtype='float64')/255  #将图像转化为数组并将像素转化到0-1之间
		# print(img_ndarray.shape)
		data[d-1]=np.ndarray.flatten(img_ndarray)    #将图像的矩阵形式保存到data中
		print("data",d,"is",data[d-1])
		d=d-1
	data = data.T
	print("data.shape:",data.shape)

	return data
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


def img2vector(filename):
	"""
	将图片数据转换为01矩阵。
	每张图片是128*128像素，也就是一共16384个字节。
	因此转换的时候，每行表示一个样本，每个样本含16384个字节。
	"""
	# 每个样本数据是1024=32*32个字节
	returnVect = zeros((1,16384))
	fr = open(filename)
	# 循环读取128行，128列。
	for i in range(128):
		lineStr = fr.readline()
		for j in range(128):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect


def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually

    return A


def train(V, r, k, e):
	m, n = shape(V)
	#先随机给定一个W、H，保证矩阵的大小
	W = mat(random.random((m, r)))
	H = mat(random.random((n, r)))

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
	linMatrix = myKNN(V.T,5)
	print("邻接矩阵：",linMatrix)
	print("邻接矩阵的规格：",linMatrix.shape)
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
	D = np.diag(np.sum(linMatrix, axis=1))
	# D = np.diag(np.sum(np.array(AW.todense()), axis=1))
	print('degree matrix:',D)
	print("度矩阵的规格：",D.shape)

	#K为迭代次数
	for x in range(k):
		#error
		V_pre = W * H.T
		E = V - V_pre
		#print E
		err = 0.0
		for i in range(m):
			for j in range(n):
				err += E[i,j] * E[i,j]
		print(err)
		data.append(err)

		if err < e:
			break
	#权值更新
		a = np.dot(V.T,W) + 100 * np.dot(linMatrix,H)
		b = np.dot(np.dot(H,W.T),W) + 100 * np.dot(D,H)
		#c = V * H.T
		#d = W * H * H.T
		for i_1 in range(n):
			for j_1 in range(r):
				if b[i_1,j_1] != 0:
					H[i_1,j_1] = H[i_1,j_1] * ( a[i_1,j_1] / b[i_1,j_1] ) 

		c = V * H
		d = W * H.T * H
		for i_2 in range(m):
			for j_2 in range(r):
				if d[i_2, j_2] != 0:
					W[i_2,j_2] = W[i_2,j_2] * ( c[i_2,j_2] / d[i_2, j_2] )

	return W,H,data


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


if __name__ == "__main__":
	R = read_data()
	# print(R)
	# N = len(R)
	# M = len(R[0])
	# K = 2
	# P = np.random.rand(N,K)
	# Q = np.random.rand(M,K)
	# nP, nQ = matrix_factorisation(R, P, Q, K)
	# nR = np.dot(np,nQ.T)
	# print(nR)



	data = []
	# R_new = []
	# A = []
	# A = np.zeros(shape=R.shape)
	# lambd = 100
	# gnmf_components = 1440
	# gnmf_itr = 100
	# neighbours = 5
	W, H ,error= train(R, 20, 4, 1e-5 )
	R_new = W * H.T
	# W, H, list_reconstruction_err_ = gnmf.gnmf(B,A, lambd,gnmf_components,max_iter=gnmf_itr)
	print("R的规格：",R.shape)
	print("W的规格：",W.shape)
	print("H的规格：",H.shape)
	print("R_new的规格：",R_new.shape)
	print(R_new)
	print(R)
	# R_pred = split_list.splitlist(np.array(R_new.ravel()))//R_new.ravel()出来的结果是一个嵌套的一维数组，按传统的分割数组的方法可以，但是1440*16384的规格太大了，电脑不行。。
	R_pred = np.array(R_new.ravel())[0]

	# n = len(error)
	# x = range(n)
	# plot(x, error, color='r', linewidth=3)
	# plt.title('Convergence curve')
	# plt.xlabel('generation')
	# plt.ylabel('loss')
	# show()
	
	# print(R_new.dtype)
	R_true = np.array(R.flatten())
	
	print(R_true)
	print(R_pred)
	print(isinstance(R_pred,list))

	result_NMI = metrics.normalized_mutual_info_score(R_true, R_pred)
	print(result_NMI)
	# result_NMI2 = NMI(R_new,R)
	# result_ACC = accuracy_score(R, R_new)
	# print(result_ACC)
	# print(result_NMI2 )

