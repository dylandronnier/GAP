import matplotlib.pyplot as plt

import numpy as np
import random

import utils 
import parser 

import variance_sparsification  
import random_sparsification  
import cur_sparsification
import hsci_sparsification
import svde_sparsification

import descriptor_utils


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

def visualize_sparsification(info, methodName='variance', full_db_mse=None):
    assert methodName in ['variance', 'random', 'cur', 'hsci']
    
    fig = plt.figure()

    extract = lambda field_name : [ info[i][field_name] for i in sorted(info)]    

    if methodName == 'variance':
        ax = fig.add_subplot(211)
        ax.set_title('Variance sparsification')
    elif methodName == 'random':
        ax = fig.add_subplot(111)
        ax.set_title('Random sparsification')
    elif methodName == 'cur':
        ax = fig.add_subplot(111)
        ax.set_title('CUR sparsification')
    elif methodName == 'hsci':
        ax = fig.add_subplot(111)
        ax.set_title('HSCI sparsification')

    ax.set_xlabel('Database size')
    ax.set_ylabel('MSE')
  
    size = extract('size_db')
    mse = extract('mse')

    ax.semilogy(size, mse)

    if full_db_mse:
        ax.axhline(y = full_db_mse, color = 'r')

    if methodName == 'variance':
        ax = fig.add_subplot(212)
        mean_var = extract('mean_joined_variance')
        ax.semilogy(size, mean_var)

        joined = extract('n_joined_cfs')
        assert np.all(np.array(joined) == joined[0])

        ax.set_xlabel('Database size')
        ax.set_ylabel('Mean variance of %d joined cfs' % joined[0])

    plt.show()

def visulalize_sparsifications_comparison(infos, names, full_db_mse=None):
    extract = lambda field_name, info : np.array([ info[i][field_name] for i in sorted(info)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Database size')
    ax.set_ylabel('MSE')
  
    for info, name in zip(infos, names):
        size = extract('size_db', info)
        mse = extract('mse', info)
        ax.semilogy(size, mse, label = name)

    if full_db_mse:
        ax.axhline(y = full_db_mse, color = 'r')

    ax.legend(loc='best')
    plt.show()

def cmp_with_random(info, methodName, start, step, iters, fulldb):
    seeds = [1, 5, 184]
    random_infos = []
    for seed in seeds:
      r_info = random_sparsification.sparsify(learn_cfs, test_cfs, descriptizers, lmbd=1e-12, sigma=1.2, startPoints=start, stepPoints=step, max_iter=iters, seed=seed)
      random_infos.append(r_info)
    
    infos = [info] + random_infos
    names = [methodName] + ['Random %d' % s for s in seeds]

    visulalize_sparsifications_comparison(infos, names, full_db_mse=fulldb)

if __name__=='__main__':
    # For reproducible results
    np.random.seed(1)
    random.seed(1)

    learn_database='BdDFluideLJ_onerho_2000'
    test_database = 'TestFluideLJ'

    take_first_learn = None
    take_first_test = None

    learn_cfs = parser.loadDatabase(learn_database, rescaleFunction=configurationRescaler, first=take_first_learn)
    test_cfs  = parser.loadDatabase(test_database,  rescaleFunction=configurationRescaler, first=take_first_test)

    descriptizers = build_descriptizers()
    full_db_mse = utils.GAP_predict(learn_cfs, test_cfs, descriptizers, lmbd=1e-12, sigma=1.2)[0]['diff_mse']
    
    # =============== bottom-top ==========
    start = 264
    step_up = 1
    iters_up = 1

    # =============== top-down ==========
    max_pts = 10
    step_down = 1
    iters_down = 3

    # =============== CUR
    epsilon = 0.01


    """
    # variance, bottom-top
    v_info = variance_sparsification.sparsify(learn_cfs, test_cfs, descriptizers, max_iter=iters_up, startPoints=start, stepPoints = step_up)	
    cmp_with_random(v_info, 'variance', start, step_up, iters_up, full_db_mse)
    """
    """
    # CUR base, cheaty bottom-top :)
    points = np.arange(start, start + iters_up * step_up, step_up)
    
    cur_info = cur_sparsification.sparsify_direct(learn_cfs, test_cfs, descriptizers, points, epsilon, seed=1)
    cmp_with_random(cur_info, 'cur base', start, step_up, iters_up, full_db_mse)
    """

    # CUR, top-bottom
    """
    cur_info = cur_sparsification.sparsify_top_bottom(learn_cfs, test_cfs, descriptizers, startPoints=max_pts, stepPoints=step_down, max_iterations=iters_down, epsilon=epsilon, lmbd=1e-12, sigma=1.2, seed=1)
    cmp_with_random(cur_info, 'cur top-bottom', max_pts - (iters_down -1 ) * step_down, step_down, iters_down, full_db_mse)
    """
    """
    HSCI_info = hsci_sparsification.pseudo_random_sparsify(learn_cfs, test_cfs, descriptizers)
    cmp_with_random(HSCI_info, 'HSCI', 50, 200, 9, full_db_mse)
    """
    """
    svde_info = svde_sparsification.sparsify_direct(learn_cfs, test_cfs, descriptizers, startPoints=max_pts, stepPoints=step_down, max_iterations=iters_down, epsilon=epsilon, lmbd=1e-12, sigma=1.2, seed=1)
    cmp_with_random(svde_info, 'svde simple', max_pts - (iters_down -1 ) * step_down, step_down, iters_down, full_db_mse)
    """

    points = np.arange(start, start + iters_up * step_up, step_up)
    
    svde_info = svde_sparsification.sparsify_direct(learn_cfs, test_cfs, descriptizers, points, epsilon, seed=1)
    # svde_info = svde_sparsification.sparsify(learn_cfs, test_cfs, descriptizers, startPoints=max_pts, stepPoints=step_down, max_iterations=iters_down, epsilon=epsilon, lmbd=1e-12, sigma=1.2, seed=1)
    cmp_with_random(svde_info, 'SVD-E simple ranking', start, step_up, iters_up, full_db_mse)





