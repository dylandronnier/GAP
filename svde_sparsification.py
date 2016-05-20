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

		database = [ learn_cfs[i] for i in svd_entropy.SVD_FS(descriptors_matrix_transposed, method='SR', r0=startPoints - stepPoints * iteration)]
		mse = utils.GAP_predict(database, test_cfs, descriptizers, log=utils.empty_printer)[0]['diff_mse']
		
		print('SVDE simple %d: mse =%e' % (len(database), mse))
		
		spars_info[iteration] = {
				'size_db' : len(database),
				'mse' : mse
		}		

		iteration += 1

	return spars_info