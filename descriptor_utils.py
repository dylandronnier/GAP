import numpy as np
import numba

from fastmath import *

@numba.autojit()
def descript_configuration(configuration, descriptizers):
	dists = __dist_compute(configuration['n_data']) 
	n_desc = len(descriptizers)
	config_desc = np.zeros(n_desc)

	for i in xrange(0, n_desc):
		assert isinstance(descriptizers[i], RadialDescriptizer)
		config_desc[i] =  np.sum(descriptizers[i].f_vectorized(dists))
		
	return config_desc

@numba.jit(nopython=True)
def __dist_compute(points3d):
	n = points3d.shape[0]
	r = np.zeros(n)
	for i in xrange(0, n):
		r[i] = norm3d(points3d[i])
	return r
	
class RadialDescriptizer(object):
	def __init__(self):
		pass

	def f(self, dist):
		assert False, "Not implemented RadialDescriptor.f"
	
	def f_vectorized(self, dists):
		assert False, "Not implemented RadialDescriptor.f_vectorized"

	def derivative(self, dist):  
		assert False, "Not implemented RadialDescriptor.derivative"

	def derivative_vectorized(self, dists):  
		assert False, "Not implemented RadialDescriptor.derivative_vectorized"

class G2Descriptizer(RadialDescriptizer):
	def __init__(self, eta, rc, rs):
		self.eta = eta
		self.rc = rc
		self.rs = rs
		
	# in general, f is not supposed to be vectorized ufunc 
	def f(self, dist):
		return g2(self.eta, self.rc, self.rs, dist)

	def f_vectorized(self, dists):
		return self.f(dists)
	
	def derivative(self, dist):  
		return g2_derivative(self.eta, self.rc, self.rs, dist)
	
	def derivative_vectorized(self, dists):
		return self.derivative(dists)  

@numba.vectorize(nopython=True)
def cutoff(cutOffDist, r):
	if r < cutOffDist:
		return 0.5 * (1.0 + np.cos(np.pi * r / cutOffDist))
	else: 
		return 0.0

@numba.vectorize(nopython=True)
def cutoff_derivative(cutOffDist, r):
	if r < cutOffDist:
	 	return -0.5 * (np.pi / cutOffDist) * np.sin(np.pi * r / cutOffDist)
  	else:
  		return 0.0

@numba.vectorize(['float64(float64, float64, float64, float64)'], nopython=True)
def g2(eta, rc, rs, r):
	return np.exp( -1.0 * eta * (r - rs) ** 2) * cutoff(rc, r)		

@numba.vectorize(['float64(float64, float64, float64, float64)'], nopython=True)
def g2_derivative(eta, rc, rs, r):
	d = cutoff_derivative(rc, r)
	c = cutoff(rc, r)
	return (d - 2.0 * eta * r * c) * np.exp(-eta * (r **2))

def produce_g2_descriptizer(eta, rc, rs):
	return G2Descriptizer(eta, rc, rs)