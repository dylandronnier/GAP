import matplotlib.pyplot as plt

import numpy as np
import random

import utils 
import parser 

import variance_sparsification  
import random_sparsification  
import cur_sparsification
import hsci_sparsification

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
	tX = np.delete(X, columnlist, 1)
	ce = svd_entropy(X) - svd_entropy(tX)
	return ce

def comp_fc(X):
	n = X.shape[1]
	ces = [comp_each_fc(X, i) for i in range(n)]
	ces = np.array(ces)
	return ces

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
	if r0 is None:
		r0 = tr0

	rindex = []
	# rX = []

	if method == 'SR':
		hcvs = [(i, ces[i]) for i in range(n)]
		hcvs = sorted(hcvs, key=lambda x:x[1], reverse=True)
		hcs = [hcvs[i][0] for i in range(n)]
		rindex = hcs[0:r0]
		# ds = hcs[r0:len(hcs)]
		# rX = np.delete(X, ds, 1)

	elif method == 'FS1':
		print("to be implemented")

	elif method == 'FS2':
		tX = np.array(X)
		# rX = tX[:, np.argmax(ces)].reshape((m, 1))
		tX = np.delete(tX, np.argmax(ces), 1)
		while tX.shape[1] < r0:
			tces = comp_fc(tX)
			rX = np.c_[rX, tX[:, np.argmax(tces)]]
			tX = np.delete(tX, np.argmax(tces), 1)

	elif method == 'BE':
		rX = np.array(X)
		while rX.shape[1] > r0:
			tces = comp_fc(rX)
			rX = np.delete(rX, np.argmin(tces), 1)

	else:
		print("to be implemented")

	# return rX, hcvs, tr0
	return rindex

if __name__=='__main__':

	np.random.seed(1)
	A = np.random.random((100,100))
	print(A.shape)
	B, hcvs, tr0 = SVD_FS(A, method='SR', r0=10)
	print(B.shape)
	print(hcvs)
	print(tr0)


      
