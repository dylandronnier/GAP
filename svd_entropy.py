import matplotlib.pyplot as plt

import numpy as np
import random

import utils 
import parser 

import variance_sparsification  
import random_sparsification  
import cur_sparsification
import hsci_sparsification

def id_manage(dic, a):
	b = len(dic)
	for i in range(a, b):
		dic[i] = dic[i] + 1
	del dic[b - 1]

def svd_entropy(X):
	rX = np.linalg.matrix_rank(X)
	ss = np.linalg.svd(X, compute_uv=False)
	s2s = ss**2
	vs = s2s / sum(s2s)
	eX = - sum(vs * np.log2(vs)) / np.log2(rX) # log10, log
	return eX

def comp_each_fc(X, i):
	tX = np.delete(X, i, 1)
	cei = svd_entropy(X) - svd_entropy(tX)
	return cei

def comp_list_fc(X, cs):
	tX = np.delete(X, cs, 1)
	ce = svd_entropy(X) - svd_entropy(tX)
	return ce

def comp_fc(X):
	n = X.shape[1]
	ces = [comp_each_fc(X, i) for i in range(n)]
	ces = np.array(ces)
	return ces

def comp_fc2(X, cs):
	n = X.shape[1]
	csl = [cs + [i] for i in range(n) if i not in cs]
	cesl = [comp_list_fc(X, c) for c in csl]
	return cesl, csl

def SVD_CS(X, method='SR', r0=None):
	
	(m, n) = X.shape
	ces = comp_fc(X)
	c = np.mean(ces)
	s = np.std(ces)
	tr0 = sum([1 for i in range(n) if ces[i] > c + s])
	if r0 is None:
		r0 = tr0
	hcvs = [(i, ces[i]) for i in range(n)]
	hcvs = sorted(hcvs, key=lambda x:x[1], reverse=True)
	hcs = [hcvs[i][0] for i in range(n)]
	rX = np.delete(X, ds, 1)
	rX = []

	return rX, hcvs, tr0

def SVD_FS(X, method='SR', r0=None):
	
	(m, n) = X.shape
	ces = comp_fc(X)
	c = np.mean(ces)
	s = np.std(ces)
	tr0 = sum([1 for i in range(n) if ces[i] > c + s])
	print(tr0)
	if r0 is None:
		r0 = tr0
		print(r0)

	rindex = []
	# rX = []

	idic  = {}
	for i in range(n):
		idic[i] = i

	if method == 'SR':
		hcvs = [(i, ces[i]) for i in range(n)]
		hcvs = sorted(hcvs, key=lambda x:x[1], reverse=True)
		hcs = [hcvs[i][0] for i in range(n)]
		rindex = hcs[0:r0]
		# ds = hcs[r0:len(hcs)]
		# rX = np.delete(X, ds, 1)

	elif method == 'FS1':
		rX = np.array(X)
		# rX = tX[:, np.argmax(ces)].reshape((m, 1))
		tn = np.argmax(ces)
		rindex.append(tn)
		while len(rindex) < r0:
			print(rindex[len(rindex) - 1])
			cesl, csl = comp_fc2(rX, rindex)
			tn = np.argmax(cesl)
			tl = csl[tn]
			rindex.append(tl[len(tl) - 1])

	elif method == 'FS2':
		tX = np.array(X)
		# rX = tX[:, np.argmax(ces)].reshape((m, 1))
		tn = np.argmax(ces)
		rindex.append(idic[tn])
		tX = np.delete(tX, tn, 1)
		id_manage(idic, tn)
		while tX.shape[1] > n - r0:
			print(tn)
			tces = comp_fc(tX)
			tn = np.argmax(tces)
			rindex.append(idic[tn])
			# rX = np.c_[rX, tX[:, np.argmax(tces)]]
			tX = np.delete(tX, tn, 1)
			id_manage(idic, tn)

	elif method == 'BE':
		rX = np.array(X)
		while rX.shape[1] > n - r0:
			tces = comp_fc(rX)
			tn = np.argmax(ces)
			rindex.append(idic[tn])
			rX = np.delete(rX, tn, 1)
			id_manage(idic, tn)

	else:
		print("to be implemented")

	# return rX, hcvs, tr0
	return rindex


if __name__=='__main__':

	np.random.seed(1)
	A = np.random.random((100,100))
	print(A.shape)
	rindex1 = SVD_FS(A, method='SR', r0=20)
	print(rindex1)
	rindex2 = SVD_FS(A, method='FS1', r0=20)
	print(rindex2)


      
