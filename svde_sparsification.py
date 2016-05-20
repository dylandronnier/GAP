import numpy as np 
import utils
import svd_entropy

# at each step we sparsify shrinked database 
def sparsify(learn_cfs, test_cfs, descriptizers, startPoints, stepPoints, max_iterations, epsilon, lmbd=1e-12, sigma=1.2, seed=1):
	np.random.seed(seed)

	spars_info = {}
	iteration = 0
	database = learn_cfs

	while iteration < max_iterations:
		descriptors_matrix_transposed = utils.descript_cfs(database, descriptizers, log=utils.empty_printer).T
		assert len(descriptors_matrix_transposed.shape) == 2

		database = [ learn_cfs[i] for i in svd_entropy.SVD_FS(descriptors_matrix_transposed, method='FS2', r0=startPoints - stepPoints * iteration)]
		# database = [ learn_cfs[i] for i in svd_entropy.SVD_FS(descriptors_matrix_transposed, method='SR', r0= 264)]
		mse = utils.GAP_predict(database, test_cfs, descriptizers, log=utils.empty_printer)[0]['diff_mse']
		
		print('SVDE simple %d: mse =%e' % (len(database), mse))
		
		spars_info[iteration] = {
				'size_db' : len(database),
				'mse' : mse
		}		

		iteration += 1

	return spars_info

def sparsify_direct(learn_cfs, test_cfs, descriptizers, n_points, epsilon, lmbd=1e-12, sigma=1.2, seed=1):
	np.random.seed(seed)

	descriptors_matrix_transposed = utils.descript_cfs(learn_cfs, descriptizers, log=utils.empty_printer).T
	assert len(descriptors_matrix_transposed.shape) == 2

	spars_info = {}

	for iteration, npts in enumerate(n_points):
		assert npts < descriptors_matrix_transposed.shape[1]

		l = [ learn_cfs[i] for i in svd_entropy.SVD_FS(descriptors_matrix_transposed, method='SR', r0=npts)]
		assert len(l) == npts

		mse = utils.GAP_predict(l, test_cfs, descriptizers, log=utils.empty_printer)[0]['diff_mse']
		print('SVDE %d:  mse = %e' % (npts, mse))

		spars_info[iteration] = {
				'size_db' : npts,
				'mse' : mse
		}

	return spars_info



