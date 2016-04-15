import parser
import descriptor_utils

import random
import time

import numpy as np
import matplotlib.pyplot as plt 

from sklearn import preprocessing
from GAP import GAP

from operator import itemgetter

random.seed(1)

def printer(s):
    print s

def build_descriptizers():
    # see g1_*, g2_* and g4_*  and SymmetricFunctionsAtoms.cpp for details
    g1s = []

    g2_coefs = [ (0.003571, 8.52, 0.0),
                 (0.03571,  8.52, 0.0),
                 (0.07142,  8.52, 0.0),
                 (0.125,    8.52, 0.0),
                 (0.21426,  8.52, 0.0),
                 (0.3571,   8.52, 0.0),
                 (0.7142,   8.52, 0.0),
                 (1.4284,   8.52, 0.0)]
	
    g2s = [descriptor_utils.produce_g2_descriptizer(eta, rc, rs) for eta, rc, rs in g2_coefs]	
    return g1s + g2s

def configurationRescaler(configuration):
    n_coords = configuration['coord'] * 1e10

    return { 'coord'   : n_coords,
             'e'       : configuration['e'] / 1.60217657e-19,
             'F'       : configuration['F'] * 1e10,
             'n_count' : configuration['n_count'],
             # TODO: introduce R[i, j] = dist(n[i], n[j]) matrix as more universal source for symmetric functions
             'n_data'  : np.array( [ r * 1e10 - n_coords for r in configuration['n_data']]) }

def descript_cfs(configurations, descriptizers, log=printer):
    log ('  build descriptors')
    tic = time.clock()
    descriptors = np.zeros( (len(configurations), len(descriptizers)) )		
    for i, cf in enumerate(configurations):
        descriptors[i, :] = descriptor_utils.descript_configuration(cf, descriptizers)
        if i > 0 and i % 1000 == 0:
            log('     descript %d' % i)
            log('    spent %r' % (time.clock() - tic))
    return descriptors

def labelize_cfs(configurations, labelizer):
    return np.array([ labelizer(cf) for cf in configurations])
	
def energy_label(cf):
    return cf['e']

def force_label(cf):
    return cf['F']
	
def format_prediction_stat(dict_with_stat):
    d = dict_with_stat
    return '\n'.join(['=' * 15 + d['title'] + '=' * 15 ,
                      '  Answer statistics:',
                      '        Mean     : %f' % d['answer_mean'],
                      '        Stddev   : %f' % d['answer_std'],
                      '        Max      : %f' % d['answer_max'],
                      '        Min      : %f' % d['answer_min'],
                      '',
                      '  Difference statistics:',
                      '        Mean: %f'     % d['diff_mean'],
                      '        Stddev: %f'   % d['diff_std'],
                      '        MSE   : %e'   % d['diff_mse'],
                      '        Max diff: %f' % d['diff_max'],
                      '=' * (30 + len(d['title']))])

def prediction_stat(predicton, answer, title="Comparison:"):
    assert predicton.shape == answer.shape
    assert len(predicton.shape) == 1
    diff = np.abs(predicton - answer)
    n = len(diff)
    mse = np.sum(diff ** 2) / n

    return { 'title'       : title,

             'answer_mean' : answer.mean(),
             'answer_std'  : answer.std(),
             'answer_max'  : answer.max(),
             'answer_min'  : answer.min(),
			
             'diff_mean'   : diff.mean(),
             'diff_std'    : diff.std(),
             'diff_mse'    : mse,
             'diff_max'    : diff.max() }

def preprocess_data(learn_cfs, test_cfs, descriptizers, log=printer):
    log('Starting descriptize and labelize (energy) learn data')
    l_desc   = descript_cfs(learn_cfs, descriptizers, log=log)
    l_lables = labelize_cfs(learn_cfs, energy_label).reshape(-1, 1)
	
    descriptors_scaler =  preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True).fit(l_desc)
    labels_scaler      =  preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True).fit(l_lables)

    l_desc, l_lables = descriptors_scaler.transform(l_desc), labels_scaler.transform(l_lables)
	
    log('Starting descriptize and labelize (energy) test data')
    t_desc   = descript_cfs(test_cfs, descriptizers, log=log)
    t_lables = labelize_cfs(test_cfs, energy_label).reshape(-1, 1) 
	
    t_desc, t_lables = descriptors_scaler.transform(t_desc), labels_scaler.transform(t_lables)

    return l_desc, l_lables, t_desc, t_lables, descriptors_scaler, labels_scaler

def test_GAP_translated(learn_cfs, test_cfs, lmbd=1e-12, sigma=1.2, log=printer):
    descriptizers = build_descriptizers()

    l_desc, l_lables, t_desc, t_lables, descriptors_scaler, labels_scaler = preprocess_data(learn_cfs, test_cfs, descriptizers, log=log)
	
    gap = GAP(descriptizers)
    gap.fit(l_desc, l_lables, lmbd, sigma, logHandler=log)

    #  =========== potentiel predicition ==============
    predicted_potentiels = np.array([ gap.predict_potentiel_by_descriptor(d) for d in t_desc]).reshape(-1, 1)
    predicted_potentiels = labels_scaler.inverse_transform(predicted_potentiels)
    # as 1d array
    predicted_potentiels = predicted_potentiels.reshape(1, -1)[0]

    real_potentels = np.array([ energy_label(c) for c in test_cfs])
    stat = prediction_stat(predicted_potentiels, real_potentels, title="Comparison of potentiel prediction")
    log(format_prediction_stat(stat))

    return stat, gap
    #  =========== potentiel predicition ==============
    
    #  =========== force predicition ==============
    
    # TODO: debug force prediction because now it is too imprecise.
    # NOTE: maybe the same imprecision happens in original C++ code, i am not sure

    """	
    predicted_forces = np.array([ GAP.predict_force_by_cf_and_desc(test_cfs[i], d) for i, d in enumerate(t_desc)])
    npier = lambda tuple_list: [ np.array(e) for e in tuple_list]
    pFx, pFy, pFz = npier(zip(*predicted_forces))
	
    real_forces = [ force_label(cf) for cf in test_cfs]
    rFx, rFy, rFz = npier(zip(*real_forces))
    
    prediction_stat(pFx, rFx, title="Comparison of Fx prediction:")	
    """
    #  =========== force predicition ==============

def select_n_cfs(cfs, n):
    total = len(cfs)
    assert n < total
    ids = range(0, total)
    random.shuffle(ids)
    return [ cfs[i] for i in ids[0:n]]

def sparsification_Variance(learn_cfs, test_cfs, max_iter=50, startPoints = 5, stepPoints =1, limit=None):
    sink = lambda s : ''

    l = select_n_cfs(learn_cfs, startPoints)

    l_desc, l_lables, t_desc, t_lables, descriptors_scaler, labels_scaler = preprocess_data(learn_cfs, test_cfs, build_descriptizers(), log=sink)
	
    iterations = 0 
    mses = []
    varns = []

    while iterations < max_iter and len(l) <= len(l_desc):
        s, gap = test_GAP_translated(l, test_cfs, log=sink)
        
        mse = s['diff_mse']
        mses.append(mse)

        print '%d samples, mse = %e, max=%e'  % (len(l), mse, s['diff_max'])
		
        var = [ (cf, gap.compute_variance(desc)) for cf, desc in zip(learn_cfs, l_desc) ]	
        var = sorted(var, key=itemgetter(1), reverse=True)

        joined_cfs, joined_vars = zip(*var[0:stepPoints])
        mean_joined_variance = np.mean(joined_vars) 
        varns.append(mean_joined_variance) 

        l += joined_cfs 
        print 'Join %d cfs with mean variance = %f ' % (len(joined_cfs), mean_joined_variance)  
        
        iterations += 1

    points_count = range(startPoints, len(l), stepPoints)

    f = plt.figure(0)
    ax = f.add_subplot(211)
    ax.set_title('MSE')
    ax.semilogy(points_count, mses)

    if limit:
        plt.axhline(y=limit, color='r')

    ax = f.add_subplot(212)
    ax.set_title('Variance')
    ax.semilogy(points_count, varns)
    plt.show()		


def norm_elementary_reduction(cfs_descriptors, cfs_lables, pointsToDelete=1, lmbd=1e-12, sigma=1.2, log=printer):
	descriptizers = build_descriptizers()

	GAP = GAPTranslated(descriptizers)
	GAP.fit(cfs_descriptors, cfs_lables, lmbd, sigma, logHandler=log)

	reduction_matrix = GAP.Kmatrix

	row_norms = [ (i, np.linalg.norm(r)) for i, r in enumerate(reduction_matrix)]
	# row_norm  high --> really similar to other points, can be deleted
	significant = sorted(row_norms, key=itemgetter(1))[0: len(row_norms) - pointsToDelete]
	assert len(cfs_descriptors) - len(significant) == pointsToDelete

	significant_ids, _ = zip(*significant)

	return significant_ids
	
def sparsification_frobenius(learn_cfs, test_cfs, lmbd=1e-12, sigma=1.2, pointsToSave=100, stepPoints=1):
	sink = lambda s : ''

	descriptizers = build_descriptizers()

	l_desc, l_lables, t_desc, t_lables, descriptors_scaler, labels_scaler = preprocess_data(learn_cfs, test_cfs, descriptizers, log=sink)

	full_db_mse = test_GAP_translated(learn_cfs, test_cfs, log=sink)[0]['diff_mse']

	mse_pts = []

	l = learn_cfs
	descs = l_desc
	lbl = l_lables

	for counter in xrange(len(learn_cfs), pointsToSave, -stepPoints):
		print 'Reduce learn db %d -> %d' % (len(l), len(l) - stepPoints)
		ids = norm_elementary_reduction(descs, lbl, pointsToDelete=stepPoints, lmbd=lmbd, sigma=sigma, log=sink)
		ids = np.array(ids)
		l     = [ l[i] for i in ids]
		descs = descs[ids]
		lbl   = lbl[ids]

		mse = test_GAP_translated(l, test_cfs, log=sink)[0]['diff_mse']
		mse_pts.append( (len(l), mse) )
		print '  mse = %e' % mse

	f = plt.figure(0)
	ax = f.add_subplot(111)
	ax.set_title('MSE')

	pts, mses = zip(*mse_pts)
	ax.semilogy(pts, mses)
	ax.invert_xaxis()
	plt.axhline(y=full_db_mse, color='r')

	plt.show()

if __name__=='__main__':
    learn_database='BdDFluideLJ_onerho_2000'
    test_database = 'TestFluideLJ'

    take_first_learn = None
    take_first_test = None

    learn_cfs = parser.loadDatabase(learn_database, rescaleFunction=configurationRescaler, first=take_first_learn)
    test_cfs  = parser.loadDatabase(test_database,  rescaleFunction=configurationRescaler, first=take_first_test)

	
    #  use simple top-bottom approach
    #sparsification_frobenius(learn_cfs, test_cfs, pointsToSave= 50, stepPoints=10)

    #  use bottom-top method using variance
    full_db_mse = test_GAP_translated(learn_cfs, test_cfs, lmbd=1e-12, sigma=1.2)[0]['diff_mse']
    sparsification_Variance(learn_cfs, test_cfs, max_iter=150, startPoints=5, stepPoints = 35, limit=full_db_mse)	
