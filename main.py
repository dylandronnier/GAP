import matplotlib.pyplot as plt

import numpy as np
import random

import utils 
import parser 

import variance_sparsification  
import random_sparsification  

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
  assert methodName in ['variance', 'random']

  fig = plt.figure()

  extract = lambda field_name : [ info[i][field_name] for i in sorted(info)]    

  if methodName == 'variance':
    ax = fig.add_subplot(211)
    ax.set_title('Variance sparsification')
  elif methodName == 'random':
    ax = fig.add_subplot(111)
    ax.set_title('Random sparsification')
  
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

def compare_sparsifications(infos, names, full_db_mse=None):
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

if __name__=='__main__':
    # For reproducible results
    random.seed(1)

    learn_database='BdDFluideLJ_onerho_2000'
    test_database = 'TestFluideLJ'

    take_first_learn = None
    take_first_test = None

    learn_cfs = parser.loadDatabase(learn_database, rescaleFunction=configurationRescaler, first=take_first_learn)
    test_cfs  = parser.loadDatabase(test_database,  rescaleFunction=configurationRescaler, first=take_first_test)

    descriptizers = build_descriptizers()
    full_db_mse = utils.GAP_predict(learn_cfs, test_cfs, descriptizers, lmbd=1e-12, sigma=1.2)[0]['diff_mse']
    
    v_info = variance_sparsification.sparsify(learn_cfs, test_cfs, descriptizers, max_iter=100, startPoints=5, stepPoints = 1)	

    seeds = [1, 2, 3]
    random_infos = []
    for seed in seeds:
      info = random_sparsification.sparsify(learn_cfs, test_cfs, descriptizers, lmbd=1e-12, sigma=1.2, startPoints=6, stepPoints=1, max_iter=100, seed=seed)
      random_infos.append(info)
    
    infos = [v_info] + random_infos
    names = ['Variance'] + ['Random %d' % s for s in seeds]
    
    compare_sparsifications(infos, names, full_db_mse)
