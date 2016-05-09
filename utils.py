import numpy as np
import time
import random

from sklearn import preprocessing
from GAP import GAP
import descriptor_utils

def default_printer(s):
    print s

def empty_printer(s):
    pass

def descript_cfs(configurations, descriptizers, log=default_printer):
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

def preprocess_data(learn_cfs, test_cfs, descriptizers, log=default_printer):
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

def GAP_predict(learn_cfs, test_cfs, descriptizers, lmbd=1e-12, sigma=1.2, log=default_printer):
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
    
def random_cfs(cfs, n):
    total = len(cfs)
    assert n < total
    ids = range(0, total)
    random.shuffle(ids)
    return [ cfs[i] for i in ids[0:n]]
