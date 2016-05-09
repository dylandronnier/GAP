import matplotlib.pyplot as plt 
import numpy as np

from operator import itemgetter
import utils 

def sparsify(learn_cfs, test_cfs, descriptizers, max_iter=50, startPoints = 5, stepPoints =1):
    sink = lambda s : ''

    l = utils.random_cfs(learn_cfs, startPoints)
    l_desc, l_lables, t_desc, t_lables, descriptors_scaler, labels_scaler = utils.preprocess_data(learn_cfs, test_cfs, descriptizers, log=sink)
	
    iteration = 0 
    spars_info = {}
    
    cur_gap_instance = utils.GAP_predict(l, test_cfs, descriptizers, log=sink)[1]

    while iteration < max_iter and len(l) <= len(l_desc):
        
        var = [ (cf, cur_gap_instance.compute_variance(desc)) for cf, desc in zip(learn_cfs, l_desc) ]	
        var = sorted(var, key=itemgetter(1), reverse=True)

        joined_cfs, joined_vars = zip(*var[0:stepPoints])
        mean_joined_variance = np.mean(joined_vars) 

        l += joined_cfs 
                
        s, cur_gap_instance = utils.GAP_predict(l, test_cfs, descriptizers, log=sink)
        mse = s['diff_mse']

        print 'Join %d cfs with mean variance = %e, mse = %e ' % (len(joined_cfs), mean_joined_variance, mse)  
        
        # Note: 
        #   size_db = old + n_joined
        #   mse = mse(size_db)
        #   mean_joined_variance computed on n_joined cfs
        #
        spars_info[iteration] = {
                'size_db' : len(l),
                'mse' : mse,
                'n_joined_cfs' : stepPoints,
                'mean_joined_variance' : mean_joined_variance
        }

        iteration += 1
    
    return spars_info