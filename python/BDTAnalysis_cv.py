#!/usr/bin/env python
"""
Script for cross validation for BDTs
"""
__author__ = "Stanislava Sevova, Elyssa Hofgard"
###############################################################################                                   
# Import libraries                                                                                                
################## 
import argparse
import sys
import os
import re
import glob
import shutil
import uproot as up
import uproot_methods
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
#ahoi.tqdm = tqdm # use notebook progress bars in ahoi
import matplotlib.pyplot as plt
import math
import glob
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
###############################################################################                                   
# Command line arguments
######################## 
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Script for running optimization for the ZH dark photon SR")
    parser.add_argument('-i',
                        '--infile',
                        dest='infile',
                        help='Input CSV file',
                        default='/afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/plotting/source/Plotting/bkgLists/all_data')
    parser.add_argument('-o',
                        '--output',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')
    
    return parser
###############################################################################                                   
    
def main(): 
    """ Run script"""
    options = getArgumentParser().parse_args()

    ### Make output dir
    dir_path = os.getcwd()
    out_dir = options.outdir
    path = os.path.join(dir_path, out_dir)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.chdir(path)

    all_data = pd.read_csv(options.infile)
    all_data = all_data[all_data['w']>0]
    #all_data['w'] = all_data['w'].abs()
    # Variables of interest
    var = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph']
    units = ['GeV','Radians','GeV','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians','']
    X = all_data
    y = all_data['event']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    
    wtrain = X_train['w']
    wtest = X_test['w']
    X_train = X_train[var]
    X_test = X_test[var]
    # Cross validation
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),learning_rate=0.05,n_estimators=400)
    param_grid = {"n_estimators": list(np.arange(375,475,1))}
    '''
    "base_estimator__max_depth": list(np.arange(1,5,1)),
    'learning_rate': [.001,.01,.02,.03,.04,.05,.06,.1,1]}
    'base_estimator__max_features': [0.5,0.6,0.7, 0.8,0.9,1.0,'sqrt'],
    'base_estimator__min_samples_leaf': [0.04, 0.06, 0.08,.1,.5,1],
    'base_estimator__min_samples_split': [2,5,10],
    'base_estimator__min_impurity_decrease': [0,.01,.05,.1]}
    'base_estimator__criterion': ['gini', 'entropy']
    'base_estimator__splitter': ['best','random']}
    'base_estimator__min_samples_split': list(np.arange(1,10,1))}
    '''
    scoring = {'AUC': 'roc_auc', 'Accuracy': metrics.make_scorer(metrics.accuracy_score), 'Recall': metrics.make_scorer(metrics.recall_score),'Balanced Accuracy': metrics.make_scorer(metrics.balanced_accuracy_score),'f1': metrics.make_scorer(metrics.f1_score)}
    sys.stdout = open('model_cv_ada_nonzero.txt','wt')
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1, cv=5,n_iter=100,scoring=scoring,refit='AUC',return_train_score=True)
    grid_result = grid.fit(X_train, y_train)
    results = grid.cv_results_
    for scorer in scoring:
        print(scorer + ":")
        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]
        best_params = results['params'][best_index]
        print(best_score)
        print(best_params)
        best_params = {k: [v] for k,v in best_params.items()}
        new_grid = RandomizedSearchCV(estimator=model,param_distributions=best_params,n_iter=1,cv=5,n_jobs=-1)
        new_grid_result = new_grid.fit(X_train,y_train)
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, new_grid_result.predict(X_test)
        print(metrics.classification_report(y_true, y_pred,sample_weight=wtest))
        print(metrics.confusion_matrix(y_test, y_pred,sample_weight=wtest))
        print()

if __name__ == '__main__':
    main()
