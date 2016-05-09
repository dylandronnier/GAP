import numpy as np 
import utils

# Of course, scipy.stats.bernoulli is more convinient, but why import all scipy for this small function?
def bernoulli(p):
	p = float(p)
	return np.random.binomial(1, p)

# note that original algorithm actually can select more or less than k columns
# see "CUR matrix decompositions for improved data analysis" of Michael W. Mahoney and Petros Drineas
# here, as a modification, we use quite straighforward solution: drop indexes with lowest <pi> value
def column_select_indexes(A, k, epsilon):
	k, epsilon = int(k), float(epsilon)
	
	U, s, V = np.linalg.svd(A)
	V = V.T

	n = A.shape[1]
	pi = np.array([np.mean(V[0:k, j] ** 2) for j in xrange(0, n)])
	assert np.abs(np.sum(pi) - 1.0) < 1e-8
	c = float(k) 
	probabilities = np.minimum(np.ones(n), c * pi)

	indexes = None 
	# loop can not be infinite, since expectance of sum(probabilities) >= k
	while True:
		indexes =  np.array([ i for i, p in enumerate(probabilities) if bernoulli(p)])
		if len(indexes) >= k:
			break

	indexes = zip(*sorted([ (i, pi[i]) for i in indexes], key=lambda v : v[1], reverse=True))[0][0:k]
	assert len(indexes) == k
	return indexes

#learn_cfs ios always used a reference for sparsification
def sparsify_direct(learn_cfs, test_cfs, descriptizers, n_points, epsilon, lmbd=1e-12, sigma=1.2, seed=1):
	np.random.seed(seed)

	descriptors_matrix_transposed = utils.descript_cfs(learn_cfs, descriptizers, log=utils.empty_printer).T
	assert len(descriptors_matrix_transposed.shape) == 2

	spars_info = {}

	for iteration, npts in enumerate(n_points):
		assert npts < descriptors_matrix_transposed.shape[1]

		l = [ learn_cfs[i] for i in column_select_indexes(descriptors_matrix_transposed, npts, epsilon)]
		assert len(l) == npts

		mse = utils.GAP_predict(l, test_cfs, descriptizers, log=utils.empty_printer)[0]['diff_mse']
		print 'CUR %d:  mse = %e' % (npts, mse)

		spars_info[iteration] = {
				'size_db' : npts,
				'mse' : mse
		}

	return spars_info


# at each step we sparsify shrinked database 
def sparsify_top_bottom(learn_cfs, test_cfs, descriptizers, startPoints, stepPoints, max_iterations, epsilon, lmbd=1e-12, sigma=1.2, seed=1):
	np.random.seed(seed)

	spars_info = {}
	iteration = 0
	database = learn_cfs

	while iteration < max_iterations:
		descriptors_matrix_transposed = utils.descript_cfs(database, descriptizers, log=utils.empty_printer).T
		assert len(descriptors_matrix_transposed.shape) == 2

		database = [ learn_cfs[i] for i in column_select_indexes(descriptors_matrix_transposed, startPoints - stepPoints * iteration, epsilon)]
		mse = utils.GAP_predict(database, test_cfs, descriptizers, log=utils.empty_printer)[0]['diff_mse']
		
		print 'CUR top-bottom %d: mse =%e' % (len(database), mse)
		
		spars_info[iteration] = {
				'size_db' : len(database),
				'mse' : mse
		}		

		iteration += 1

	return spars_info
