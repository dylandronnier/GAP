import numpy as np
import utils

def HSCI(x, y, k, l):
    """ Compute HSCI(x, y) whith k which is the kernel of the input space
    and l which is the kernel of the output space """
    m = x.shape[0]
    assert(y.shape[0] == m)
    ii = np.ones((m,1))
    K = np.array([[k(x[i,:], x[j,:]) if i != j else 0. for i in range(m)] for j in range(m)])
    L =np.array([[l(y[i,:], y[j,:]) if i != j else 0. for i in range(m)] for j in range(m)])
    KL = np.dot(K, L)
    return float(KL.trace() + reduce(np.dot, [ii.T, K, ii, ii.T, L, ii])/(m-1)/(m-2) - 2 * ii.T.dot(KL).dot(ii)/(m-2))/(m-3)/m


def sparsify(learn_cfs, test_cfs, max_iter=50, stepPoints =1, limit=None):
    sink = lambda s : ''
    
    descriptizers = build_descriptizers()
    
    l_desc, l_labels, t_desc, t_lables, descriptors_scaler, labels_scaler = preprocess_data(learn_cfs, test_cfs, descriptizers, log=sink)
    
    full_db_mse = test_GAP_translated(learn_cfs, test_cfs, log=sink)[0]['diff_mse']

    mse_pts = []

    l = learn_cfs
    descs = l_desc
    lbl = l_labels
    
    #print descs.shape[0], lbl.shape[0]
    for i in range(2000):
        b = np.array([j != i for j in range(2000)])
        print HSCI(descs[b], lbl[b], lambda xi, xj : np.exp(-np.dot(xj-xi, xj-xi)), lambda yi, yj :  np.exp(-np.dot(yj-yi, yj-yi)))
