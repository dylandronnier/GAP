import numpy as np
import utils
import functools

def HSCI(x, y, k, l):
    """ Compute HSCI(x, y) whith k which is the kernel of the input space
    and l which is the kernel of the output space """
    m = x.shape[0]
    assert(y.shape[0] == m)
    ii = np.ones((m,1))
    K = np.array([[k(x[i,:], x[j,:]) if i != j else 0. for i in range(m)] for j in range(m)])
    L =np.array([[l(y[i,:], y[j,:]) if i != j else 0. for i in range(m)] for j in range(m)])
    KL = np.dot(K, L)
    return float(KL.trace() + functools.reduce(np.dot, [ii.T, K, ii, ii.T, L, ii])/(m-1)/(m-2) - 2 * ii.T.dot(KL).dot(ii)/(m-2))/(m-3)/m


def pseudo_random_sparsify(learn_cfs, test_cfs, descriptizers,
                           max_iter=150, prob=0.1, limit=250):
    """ Naive sparsification with Hilbert-Schmidt indepedance criteria """

    spars_info = {}
    
    #print descs.shape[0], lbl.shape[0]
    var = float('inf')
    selection = np.ones(2000, dtype = bool)
    subselection = np.ones(2000, dtype = bool)
    kernel = lambda xi, xj : np.exp(-np.dot(xj-xi, xj-xi))
    iterations = 0
    desc, lbl, t_desc, t_lables, descriptors_scaler, labels_scaler = utils.preprocess_data(learn_cfs, test_cfs, descriptizers)

    while(selection.sum() > limit):
        hs = 0.0
        for i in range(max_iter):
            b = np.logical_and(np.random.binomial(1, prob, 2000), selection)
            var = HSCI(desc[b,:], lbl[b], kernel, kernel)
            print(var)
            if hs < var :
                hs = var
                subselection = b
        selection = np.logical_and(selection, subselection)
        lcfs = list( learn_cfs[i] for i in selection.nonzero()[0] )
        mse = utils.GAP_predict(lcfs, test_cfs, descriptizers, 
                                log=utils.empty_printer)[0]['diff_mse']
        print('Iterations %d : Taille de la selection : %d ; HSCI = %e ; mse = %e' % (iterations, selection.sum(), hs, mse))

        spars_info[iterations] = {
            'size_db' : selection.sum(),
            'mse' : mse
        }
        iterations += 1

    return spars_info

def pseudo_random_sparsify2(learn_cfs, test_cfs, descriptizers,
                           max_iter=150, prob=0.1, limit=250):
    """ Naive sparsification with Hilbert-Schmidt indepedance criteria """

    spars_info = {}
    
    #print descs.shape[0], lbl.shape[0]
    var = float('inf')
    selection = np.zeros(2000, dtype = bool)
    kernel = lambda xi, xj : np.exp(-np.dot(xj-xi, xj-xi))
    iterations = 0
    desc, lbl, t_desc, t_lables, descriptors_scaler, labels_scaler = utils.preprocess_data(learn_cfs, test_cfs, descriptizers)

    while(selection.sum() < limit):
        hs = 0.0
        for i in range(max_iter):
            b = np.logical_and(np.random.binomial(1, prob, 2000), selection)
            var = HSCI(desc[b,:], lbl[b], kernel, kernel)
            print(var)
            if hs < var :
                hs = var
                subselection = b
        selection = np.logical_or(selection, subselection)
        lcfs = list( learn_cfs[i] for i in selection.nonzero()[0] )
        mse = utils.GAP_predict(lcfs, test_cfs, descriptizers, 
                                log=utils.empty_printer)[0]['diff_mse']
        print('Iterations %d : Taille de la selection : %d ; HSCI = %e ; mse = %e' % (iterations, selection.sum(), hs, mse))

        spars_info[iterations] = {
            'size_db' : selection.sum(),
            'mse' : mse
        }
        iterations += 1

    return spars_info

if __name__=='__main__':
    """ TEST DE HSCI """
    x = np.ones((200,8))
    y = np.zeros((200,1))
    k = lambda xi, xj : np.exp(-np.dot(xj-xi, xj-xi))
    print(HSCI(x, y, k, k))
                
    
