import numpy as np 
import numba

from fastmath import inner_product, norm3d

def default_printer(s):
    print(s)

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
	
    def compute_variance(self, cf_desc):
        """ uncertainty of the prediction """
        return cf_variance(cf_desc, self.Cinv, self.train_data, self.c, self.lambdac)
		
@numba.autojit()
def cf_variance(test_configuration_desc, Cinv, train_cfs_descriptors, c, lambdac):
    v = 1.0 + len(train_cfs_descriptors) * lambdac 
    kx = Kx_xi(test_configuration_desc, train_cfs_descriptors, c)
    v -= inner_product(kx, Cinv.dot(kx))
    if v < 0.0:
        v = 0.0
    return v

@numba.jit(nopython=True)
def Kernel(x, y, c):
    return np.exp( - c * inner_product(x - y, x - y))

@numba.jit(nopython=True)
def Kx_xi(predict_descriptor, train_descriptors, c):
    n = len(train_descriptors)
    r = np.zeros(n)
    
    for i in range(0, n):
        r[i] = Kernel(predict_descriptor, train_descriptors[i], c)
    return r

@numba.jit(nopython=True)
def cf_potentiel(test_configuration_desc, alpha, train_cfs_descriptors, c):
    kx = Kx_xi(test_configuration_desc, train_cfs_descriptors, c)
    return inner_product(kx, alpha)

@numba.jit(nopython=True)
def Kmat(descriptors, c):
    n = len(descriptors)
    K = np.zeros( (n, n))

    for i in range(0, n):
        for j in range(0, n):
            K[i][j] = Kernel(descriptors[i], descriptors[j], c)
    return K
