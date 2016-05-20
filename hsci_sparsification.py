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
            print var
            if hs < var :
                hs = var
                subselection = b
        selection = np.logical_and(selection, subselection)
        lcfs = list( learn_cfs[i] for i in selection.nonzero()[0] )
        mse = utils.GAP_predict(lcfs, test_cfs, descriptizers, 
                                log=utils.empty_printer)[0]['diff_mse']
        print 'Iterations %d : Taille de la selection : %d ; HSCI = %e ; mse = %e' % (iterations, selection.sum(), hs, mse)

        spars_info[iterations] = {
            'size_db' : selection.sum(),
            'mse' : mse
        }
        iterations += 1

    return spars_info

def sparsifyFOHSIC(learn_cfs, test_cfs, descriptizers, limit=250, max_iter=[2400, 1200, 600, 300, 200, 100, 100, 100, 100, 80, 80, 60, 50, 40, 40]):
    """ Naive sparsification with Hilbert-Schmidt indepedance criteria """

    spars_info = {}
    
    #print descs.shape[0], lbl.shape[0]
    var = float('inf')
    #selection = np.random.binomial(1, 0.005, 2000)
    selection = np.zeros(2000, dtype = bool)
    subselection = np.zeros(2000, dtype = bool)
    kernel = lambda xi, xj : np.exp(-np.dot(xj-xi, xj-xi))
    iterations = 0
    desc, lbl, t_desc, t_lables, descriptors_scaler, labels_scaler = utils.preprocess_data(learn_cfs, test_cfs, descriptizers)
    
    """
    K = np.array([[kernel(desc[i,:], desc[j,:]) if i != j else 0. for i in range(2000)] for j in range(2000)])
    L = np.array([[kernel(lbl[i,:], lbl[j,:]) if i != j else 0. for i in range(2000)] for j in range(2000)])
    KL = K*L
    Cub = np.array([[[K[i,k]*L[k,j] for i in range(2000)] for j in range(2000)] for k in range(2000)])
    

    lcfs = list( learn_cfs[i] for i in selection.nonzero()[0] )
    mse = utils.GAP_predict(lcfs, test_cfs, descriptizers, 
                            log=utils.empty_printer)[0]['diff_mse']
    #print 'Iterations %d : Taille de la selection : %d ; HSCI = %e ; mse = %e' % (iterations, selection.sum(), hs, mse)

    spars_info[iterations] = {
        'size_db' : selection.sum(),
        'mse' : mse
    }
    iterations += 1
    """

    while(selection.sum() < limit):
        hs = 0.0
        #for i in range(2000):
        if iterations >= len(max_iter):
            repet = 30
        else:
            repet = max_iter[iterations]
        for i in range(repet):
            b = np.logical_or(np.random.binomial(1, 0.01, 2000), selection)
            #b[i] = True
            var = HSCI(desc[b,:], lbl[b], kernel, kernel)
            print var
            #m = selection.sum()
            #print 'HSCI 2 :'
            #var2 = (KL[b,:][:,b].sum() + K[b,:][:,b].sum()*L[b,:][:,b].sum()/(m-1)/(m-2) - 2*Cub[b,:,:][:,b,:][:,:,b].sum()/(m-2))/(m-3)/m
            #print var2
            if hs < var :
                hs = var
                subselection = b
        selection = subselection
        lcfs = list( learn_cfs[i] for i in selection.nonzero()[0] )
        mse = utils.GAP_predict(lcfs, test_cfs, descriptizers, 
                                log=utils.empty_printer)[0]['diff_mse']
        print 'Iterations %d : Taille de la selection : %d ; HSIC = %e ; mse = %e' % (iterations, selection.sum(), hs, mse)

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
    print HSCI(x, y, k, k)
                
    
