from __future__ import print_function

import time
import os.path
from itertools import product
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd
from platypus.algorithms import *
from sklearn.model_selection import train_test_split

from skmoefs.rcs import RCSInitializer, RCSVariator
from skmoefs.fuzzysets.TriangularFuzzySet import TriangularFuzzySet
from skmoefs.fuzzysets.SingletonFuzzySet import SingletonFuzzySet
from skmoefs.fuzzysets.UniverseFuzzySet import UniverseFuzzySet
from skmoefs.fuzzysets.TrapezoidalFuzzySet import TrapezoidalFuzzySet
from skmoefs.discretization.discretizer_base import fuzzyDiscretization
from skmoefs.discretization.discretizer_crisp import CrispMDLFilter
from skmoefs.discretization.discretizer_fuzzy import FuzzyMDLFilter
from skmoefs.toolbox import MPAES_RCS, load_dataset, normalize, is_object_present, store_object, load_object, milestones


def test_time():
    n = 30
    intervals = np.zeros([n])
    for seed in range(n):
        start_time = time.time()
        test_fit('iris', 'mpaes22', 0)
        intervals[seed] = time.time() - start_time
    print('Mean processing time is ' + np.mean(intervals) + ' seconds')


def set_rng(seed):
    np.random.seed(seed)
    random.seed(seed)


def make_directory(path):
    try:
        os.stat(path)
    except:
        os.makedirs(path)


def test_fit(dataset, alg, seed, nEvals=50000, store=True):
    path = 'results/' + dataset + '/' + alg + '/'
    make_directory(path)
    set_rng(seed)
    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X_n, y_n = normalize(X, y, attributes)

    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3, random_state=seed)

    Amin = 1
    M = 50
    capacity = 32
    divisions = 8
    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=5, method='uniform')
    initializer = RCSInitializer(discretizer=discretizer)
    if store:
        base = path + 'moefs_' + str(seed)
        if not is_object_present(base):
            mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                      divisions=divisions, variator=variator,
                                      initializer=initializer, moea_type=alg,
                                      objectives=['accuracy', 'trl'])
            mpaes_rcs_fdt.fit(Xtr, ytr, max_evals=nEvals)
            store_object(mpaes_rcs_fdt, base)
        else:
            mpaes_rcs_fdt = load_object(base)
    else:
        mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                  divisions=divisions, variator=variator,
                                  initializer=initializer, moea_type=alg,
                                  objectives=['accuracy', 'trl'])
        mpaes_rcs_fdt.fit(Xtr, ytr, max_evals=nEvals)

    mpaes_rcs_fdt.show_pareto(Xte, yte)


def test_crossval(dataset, alg, seed, nEvals=50000):
    path = 'results/' + dataset + '/' + alg + '/'
    make_directory(path)
    set_rng(seed)
    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X, y = normalize(X, y, attributes)

    Amin = 1
    M = 100
    capacity = 64
    divisions = 8
    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=5, method='uniform')
    initializer = RCSInitializer(discretizer=discretizer)
    mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                              divisions=divisions, variator=variator, initializer=initializer)
    return mpaes_rcs_fdt.cross_val_score(X, y, nEvals=nEvals, num_fold=5, seed=seed,
                                         storePath=path + str(seed))


def test_multiple():
    """
    Cross-validate different datasets with multiple algorithms
    """
    datasets = ['appendicitis', 'bupa', 'glass', 'haberman',
                'hayes-roth', 'iris', 'pima', 'saheart', 'vehicle', 'wine']
    algs = ['mpaes22', 'nsga3', 'moead']
    seeds = (np.arange(6) + 1)
    # Define a pool of Threads. Each one will cross-validate a specific tuple (dataset, algorithm, seed)
    pool = Pool(6)
    results = []
    r = pool.starmap_async(test_crossval, product(datasets, algs, seeds), callback=results.append)
    r.wait()  # Wait on the results


def parse_results():
    """
    Retrieve and store statistics on cross-validation
    """
    datasets = ['appendicitis', 'bupa', 'glass', 'haberman',
                'hayes-roth', 'iris', 'pima', 'saheart', 'vehicle', 'wine']
    algs = ['mpaes22', 'nsga3', 'moead']
    seeds = (np.arange(6) + 1)

    measurements = ['Acc_tr', 'Acc_ts', 'TRL', 'NRules', 'Precision', 'Recall', 'Fscore']
    positions = ['first', 'median', 'last']
    cols = ['alg', 'dataset', 'milestone'] + [t[0] + '_' + t[1] for t in itertools.product(measurements, positions)]

    stats = {k: [] for k in cols}
    stats_arch = {k: [] for k in cols}
    for alg in algs:
        for dataset in datasets:
            data = {k: np.zeros(30) for k in cols}
            data_arch = {k: np.full([len(milestones), 30], np.nan) for k in cols}
            for k in range(len(seeds)):
                set_rng(seeds[k])
                scores, scores_archives = test_crossval(dataset, alg, seeds[k])
                for fold in range(5):
                    for i, position in enumerate(positions):
                        for j, meas in enumerate(measurements):
                            data[meas + '_' + position][k * 5 + fold] = scores[j, i, fold]
                            for s in range(len(scores_archives)):
                                data_arch[meas + '_' + position][s][k * 5 + fold] = scores_archives[s][j, i, fold]

            # Build dataframes with mean values throughout folds/seeds
            for s in range(8):
                if s == 0:
                    stats['alg'].append(alg)
                    stats['dataset'].append(dataset)
                    stats['milestone'].append(-1)
                stats_arch['alg'].append(alg)
                stats_arch['dataset'].append(dataset)
                stats_arch['milestone'].append(milestones[s])
                for k in cols:
                    if k != 'alg' and k != 'dataset' and k != 'milestone':
                        if s == 0:
                            stats[k].append(np.nanmean(data[k]))
                        stats_arch[k].append(np.nanmean(data_arch[k][s, :]))
    df = pd.DataFrame(stats, columns=cols)
    df_arch = pd.DataFrame(stats_arch, columns=cols)
    df.to_csv('csv/stats.csv', mode='a', header=True)
    df_arch.to_csv('csv/stats_archives.csv', mode='a', header=True)


if __name__ == "__main__":
    ###########
    # FUZZYSETS
    ###########

    # Fuzzy Universe

    # Fuzzy Singleton
    fuzzy_singleton = SingletonFuzzySet.createFuzzySet(43.2, 3)
    print(fuzzy_singleton)

    fuzzy_singletons = SingletonFuzzySet.createFuzzySets([17.5, 24.3, 11.4, 23.1])
    for fuzzy_singleton in fuzzy_singletons:
        print(fuzzy_singleton)

    # Fuzzy Triangular Set
    triangular_fuzzy_params = [0.02777778, 0.27083333, 0.51388889, 0.75694444, 1.]
    fuzzy_triangulars = TriangularFuzzySet.createFuzzySets(triangular_fuzzy_params, True)
    for fuzzy_triangular in fuzzy_triangulars:
        print(fuzzy_triangular)

    # Fuzzy Trapezoidal Set
    trapezoidal_fuzzy_params = [0.02777778, 0.27083333, 0.51388889]
    trpzPrm = 0.56
    fuzzy_trapezoidal = TrapezoidalFuzzySet.createFuzzySet(trapezoidal_fuzzy_params, trpzPrm)
    print(fuzzy_trapezoidal)

    trapezoidal_fuzzy_params = [0.02777778, 0.27083333, 0.51388889, 0.12345678]
    trpzPrm = 0.56
    fuzzy_trapezoidals = TrapezoidalFuzzySet.createFuzzySets(trapezoidal_fuzzy_params, trpzPrm, True)
    for fuzzy_trapezoidal in fuzzy_trapezoidals:
        print(fuzzy_trapezoidal)


    #########
    # TOOLBOX
    #########

    # load dataset
    X, y, attributes, inputs, outputs = load_dataset("iris")
    print(attributes)
    print(inputs)
    print(outputs)
    print(X)
    print(y)
    print(type(attributes))
    print(type(inputs))
    print(type(outputs))
    print(type(X))
    print(type(y))

    # normalize dataset
    X_n, y_n = normalize(X, y, attributes)
    print(X_n)
    print(y_n)


    ##############
    # DISCRETIZERS
    ##############
    fuzzy_discretizer_uniform = fuzzyDiscretization(numSet=5, method='uniform')
    fuzzy_splits_uniform = fuzzy_discretizer_uniform.run(X_n, [True, True, True, True])
    print("Fuzzy Discretizer - equifreq:\n" + str(fuzzy_splits_uniform))

    fuzzy_discretizer_equifreq = fuzzyDiscretization(numSet=5, method='equifreq')
    fuzzy_splits_equifreq = fuzzy_discretizer_equifreq.run(X_n, [True, True, True, True])
    print("Fuzzy Discretizer - equifreq:\n" + str(fuzzy_splits_equifreq))

    crisp_mdlf_discretizer = CrispMDLFilter(3, X_n, y_n, [True, True, True, True])
    crisp_mdlf_splits = crisp_mdlf_discretizer.run()
    print("Crisp MDLF Discretizer:\n" + str(crisp_mdlf_splits))

    fuzzy_mdlf_discretizer = FuzzyMDLFilter(3, X_n, y_n, [True, True, True, True])
    fuzzy_mdlf_splits = fuzzy_mdlf_discretizer.run()
    print("Fuzzy MDLF Discretizer:\n" + str(fuzzy_mdlf_splits))


    #####
    # RCS
    #####
    rcs_initializer = RCSInitializer()


    #freeze_support()
    #test_multiple()
    #parse_results()
    #test_crossval('appendicitis', 'moead', 1, nEvals=1000)
    test_fit('iris', 'mpaes22', 2, nEvals=2000, store=False)