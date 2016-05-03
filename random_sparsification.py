import numpy as np
import utils

import random

def sparsify(learn_cfs, test_cfs, descriptizers, lmbd=1e-12, sigma=1.2, startPoints=5, stepPoints=1, max_iter=50, seed=1):
	random.seed(seed)

	sink = lambda s : ''

	indexes_shuffled= range(0, len(learn_cfs))
	random.shuffle(indexes_shuffled)

	spars_info = {}

	for iteration in xrange(0, max_iter):
		npts = startPoints + iteration * stepPoints
		assert npts < len(learn_cfs)
		live_indexes = indexes_shuffled[0 : npts]

		l = [ learn_cfs[i] for i in live_indexes]
		mse = utils.GAP_predict(l, test_cfs, descriptizers, log=sink)[0]['diff_mse']
		print '  Random points (%d), mse = %e' % (len(l), mse)

		spars_info[iteration] = {
				'size_db' : npts,
				'mse' : mse,
				'n_joined_cfs' : stepPoints
		}

	return spars_info
