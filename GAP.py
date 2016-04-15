import numpy as np 
import numba
from fastmath import inner_product, norm3d

def default_printer(s):
    print s

class GAP(object):
    """ Gaussian Approximation Potential """
    def __init__(self, descriptizers):
        self.descriptizers = descriptizers
  
    def fit(self, data, y, lambdac, sigma, logHandler = default_printer):
        """ Fit potential thanks to Gaussian Process Regression """
        self.c = 1.0 / (2.0 * (sigma **2) )
        self.lambdac = lambdac
        self.train_data = data

        logHandler('GAP translated fitting')
        logHandler('    lambda = %e' % lambdac)
        logHandler('    sigma  = %f' % sigma)
        logHandler('    c      = %f' % self.c)

        n = self.train_data.shape[0]		
        I = np.eye (n)
                
        self.Kmatrix = Kmat(self.train_data, self.c)
        logHandler('    Built K')
	
        C = self.Kmatrix + self.lambdac * n * I
        logHandler('    Got C by regularization of K')
	
        self.Cinv = np.linalg.inv(C)	
        logHandler('    Inversed C')
	
        self.alpha = self.Cinv.dot(y).reshape(1, -1)[0]
        
        logHandler('Finished fitting')

    def predict_potentiel_by_descriptor(self, cf_desc):
        """ Approximation of the potential after learning step """
        return cf_potentiel(cf_desc, self.alpha, self.train_data, self.c)
	
    #def predict_force_by_cf_and_desc(self, cf, cf_desc):
    #    """ Force prediction """
    #    derivs = derivatives(cf['n_data'], self.descriptizers)
    #    return self.__compute_force(cf_desc, derivs) 
	
    #def __compute_force(self, cf_desc, cf_derivatives):
    #    return cf_forces(cf_derivatives, cf_desc, self.alpha, self.train_data, self.c)	

    def compute_variance(self, cf_desc):
        """ uncertainty of the prediction """
        return cf_variance(cf_desc, self.Cinv, self.train_data, self.c, self.lambdac)
		
#@numba.jit(nopython=True)
def cf_variance(test_configuration_desc, Cinv, train_cfs_descriptors, c, lambdac):
    v = 1.0 + len(train_cfs_descriptors) * lambdac 
    kx = Kx_xi(test_configuration_desc, train_cfs_descriptors, c)
    v -= inner_product(kx, Cinv.dot(kx))
    if v < 0.0:
        v = 0.0
    return v

"""
@numba.autojit()
def derivatives(configuration_centered_neighbours, descriptors):
    assert configuration_centered_neighbours.shape[1] == 3
    
    neighbours = configuration_centered_neighbours.shape[0]
    n_descriptors = len(descriptors)
	
    der = np.zeros( (n_descriptors, 3 ))

    for descid in xrange(0, n_descriptors):
        for xyz in configuration_centered_neighbours:
            dist = norm3d(xyz)
            d = descriptors[descid].derivative(dist)
            for k in xrange(0, 3):
                der[descid][k] += d * xyz[k] / dist 
    return der
"""

@numba.jit(nopython=True)
def Kernel(x, y, c):
    v =  np.exp( - c * inner_product(x - y, x - y))
    return v

@numba.jit(nopython=True)
def Kx_xi(predict_descriptor, train_descriptors, c):
    n = len(train_descriptors)
    r = np.zeros(n)
    
    for i in xrange(0, n):
        r[i] = Kernel(predict_descriptor, train_descriptors[i], c)
    return r

"""
@numba.jit(nopython=True)
def cf_forces(cf_derivs, test_configuration_desc, alpha, train_cfs_descriptors, c):	 
   n = len(alpha) # number of train points
   m = cf_derivs.shape[0] # number of descriptors
        
    assert train_cfs_descriptors.shape[0] == n
    assert train_cfs_descriptors.shape[1] == m

    corr = Kx_xi(test_configuration_desc, train_cfs_descriptors, c) 
	
    derivees = np.zeros(3)

    for k in xrange(0, 3):
        for i in xrange(0, n):
            v = 0.0
            for p in xrange(0, m):
                v += -(test_configuration_desc[p] - train_cfs_descriptors[i][p]) * cf_derivs[p][k]		
                derivees[k] += alpha[i] * corr[i] * v

            derivees[k] *= 2 * c
  
    return derivees
"""

@numba.jit(nopython=True)
def cf_potentiel(test_configuration_desc, alpha, train_cfs_descriptors, c):
    kx = Kx_xi(test_configuration_desc, train_cfs_descriptors, c)
    return inner_product(kx, alpha)

@numba.jit(nopython=True)
def Kmat(descriptors, c):
    n = len(descriptors)
    K = np.zeros( (n, n))

    for i in xrange(0, n):
        for j in xrange(0, n):
            K[i][j] = Kernel(descriptors[i], descriptors[j], c)
    return K
