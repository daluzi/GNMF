def splitlist(list):
	alist = []
	a = 0
	for sublist in list:
		try: #用try来判断是列表中的元素是不是可迭代的，可以迭代的继续迭代
			for i in sublist:
				alist.append (i)
		except TypeError: #不能迭代的就是直接取出放入alist
			alist.append(sublist)
	for i in alist:
		if type(i) == type([]):#判断是否还有列表
			a =+ 1
			break
	if a==1:
		return splitlist(alist) #还有列表，进行递归
	if a==0:
		return alist