# import necessary packages -----------------------------------------------------------------------
import numpy as np 
import pandas as pd
from time import time
from os import path, mkdir

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI

# argument parsing --------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('results_name', type=str)
parser.add_argument('--query_steps', type=int, default=80)
parser.add_argument('--test_run', type=bool, default=False)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--safety_factor', type=float, default=1.0)

args = parser.parse_args()

results_name = args.results_name
query_steps = args.query_steps
test_run = args.test_run
n_restarts = args.n_restarts
safety_factor = args.safety_factor

# model list --------------------------------------------------------------------------------------
model_list = [
    ['min', 'Abs', 'LPS'],
    ['min', 'Abs', 'MPLA'],
    ['min', 'Abs', 'CpG'],
    ['min', 'Abs', 'Gen3'],
    ['max', 'Abs', 'LPS'],
    ['max', 'Abs', 'MPLA'],
    ['max', 'Abs', 'CpG'],
    ['max', 'Abs', 'Gen3'],
    ['max', 'Lum', 'LPS'],
    ['max', 'Lum', 'MPLA'],
    ['max', 'Lum', 'CpG'],
    ['max', 'Lum', 'Gen3'],
    ['max', 'Lum', 'cGAMP'],
    ['max', 'Lum', 'Gen4']
]

# data loading (X and Y) --------------------------------------------------------------------------
if test_run:
    train_size = 10
    pool_size = 100
    X_train = np.load('../_Data/X_train.npy')[:train_size]
    X_pool = np.load('../_Data/X_pool_CSC_10mM.npy')[:pool_size]
    X_pool_idx = np.arange(pool_size)

    Y_df = pd.read_csv('../_Data/Prim_Agn4_Gen3Gen4.csv')[:train_size]

    query_steps = int(pool_size / len(model_list))

else:
    X_train = np.load('../_Data/X_train.npy')
    X_pool = np.load('../_Data/X_pool_CSC_10mM.npy')
    X_pool_idx = np.arange(X_pool.shape[0])

    Y_df = pd.read_csv('../_Data/Prim_Agn4_Gen3Gen4.csv')

# helper functions --------------------------------------------------------------------------------
def time_hms(start_point):
    t = time() - start_point
    # covert time (in seconds) to h:m:s format
    h = '%02d' % int(t // 3600)
    m = '%02d' % int(t % 3600 // 60)
    s = '%02d' % int(t % 60)
    return  f'{h}:{m}:{s}'

# GPR models starting -----------------------------------------------------------------------------
all_start = time()

optimizer_list = []
for model in model_list:

    model_start = time()

    maxOrmin = model[0]
    AbsOrLum = model[1]
    Agonist = model[2]

    y_train = Y_df[f'{Agonist} {AbsOrLum}'].to_numpy().reshape((-1, 1)) # experimental mean
    y_err = Y_df[f'{Agonist} {AbsOrLum} Std'].to_numpy().mean() # experimental std
    y_err = safety_factor * y_err # safety factor

    # if minimizing, reverse Y
    if maxOrmin == 'min':
        y_train = - y_train

    # defining GPR
    regressor = GaussianProcessRegressor(kernel=1.0 * kernels.RBF(length_scale=1.0), 
                                        n_restarts_optimizer=n_restarts,
                                        alpha=y_err, 
                                        normalize_y=True)
    optimizer = BayesianOptimizer(estimator=regressor,
                                  X_training=X_train, 
                                  y_training=y_train,
                                  query_strategy=max_EI)
    optimizer_list.append(optimizer)

    # predicting over all points
    y_pred, y_std = optimizer.predict(X_pool, return_std=True)

    # save GPR results ----------------------------------------------------------------------------
    if not test_run:
        model_path = f'{maxOrmin}_{AbsOrLum}_{Agonist}'
        if not path.exists(model_path):
            mkdir(model_path)
        np.save(f'{model_path}/y_train.npy', y_train)
        np.save(f'{model_path}/y_err.npy', y_err)
        np.save(f'{model_path}/y_pred.npy', y_pred)
        np.save(f'{model_path}/y_std.npy', y_std)



# Bayesian optimization by Kriging believer -------------------------------------------------------
results = []
total_num_unique = 0
for step in range(query_steps):

    # time marker
    step_start = time()

    # dict to save result in current step
    current_result = {'step': step + 1}

    # select unqueried entry from the full pool
    X_query = X_pool[X_pool_idx]

    # querying of each model (14 models)
    query_idx = [optimizer.query(X_query)[0][0] for optimizer in optimizer_list]
    for id_model in range(len(model_list)):
        model_name = ' '.join(model_list[id_model])
        current_result[model_name] = X_pool_idx[query_idx[id_model]]

    # remove duplicates
    query_idx = list(dict.fromkeys(query_idx))
    total_num_unique += len(query_idx)
    current_result['total num unique'] = total_num_unique

    # remove queried candidates from pool
    # the pool itself is not modified
    # I use a list of indices to track if each entry selected or not
    X_pool_idx = np.delete(X_pool_idx, query_idx, axis=0)

    # predict selected query point for purposes of Kriging believer
    # suppose "measured" value = predicted value
    # this is performed for each model using all unique points from 14 models
    X_KB = X_query[query_idx, :]
    for optimizer in optimizer_list:
        y_KB = optimizer.predict(X_KB, return_std=False)
        optimizer.teach(X_KB, y_KB)

    # bookkeeping
    current_result['elapsed time'] = time_hms(step_start)
    current_result['total elapsed time'] = time_hms(all_start)
    results.append(current_result)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{results_name}.csv', sep=',', index=False)
    results_df.to_csv(f'{results_name}.tsv', sep='\t', index=False)