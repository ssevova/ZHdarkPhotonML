#!/usr/bin/env python
"""
Script to test different combinations of variables for BDT analysis. Takes in a list of variables and explores removing one variable at a time.
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
                        default = '/afs/cern.ch/user/e/ehofgard/public/data/all_data')
    parser.add_argument('-o',
                        '--output',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')
    parser.add_argument('-v',
                        '--variables',
                        dest='variables',
                        help = 'List of variables to consider',
                        default = '/afs/cern.ch/user/e/ehofgard/public/ZHdarkPhoton/source/Plotting/HInvPlot/macros/mva_analysis/vars.txt')
    
    return parser
###############################################################################                                   

def compare_train_test(clf, X_train, y_train, X_test, y_test, name,bins=30):
    plt.clf()
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled',density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled',density=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    #plt.xlim(-10,10)
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    plt.title("Decision Scores Left Out " + name)
    plt.xlabel("Score")
    plt.legend(loc='best')
    plt.savefig("BDT_decision_"+name+".pdf")

def plot_roc(fpr,tpr,roc_auc):
    plt.clf()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("roc.pdf")

def plot_probabilities(probas,y_pred,name):
    plt.clf()
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(0,k)
        plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(3, 9, 1, 5), origin='lower')
        plt.xticks(())
        plt.yticks(())
        idx = (y_pred == k)
        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

def plot_probs(y_test,probs,model):
    prediction = probs[:,1]
    arr_true_0_indices = (y_test == 0.0)
    arr_true_1_indices = (y_test == 1.0)
    plt.clf()
    arr_pred_0 = prediction[arr_true_0_indices]
    arr_pred_1 = prediction[arr_true_1_indices]

    plt.hist(arr_pred_0, bins=40, label='Background', density=True, histtype='step')
    plt.hist(arr_pred_1, bins=40, label='Signal', density=True, histtype='step')
    plt.xlabel('Probability of being Positive Class')
    plt.legend()
    plt.title('Output Test Probabilities Left Out ' + model)
    plt.savefig('output_prob_' + model + ".pdf")

def plot_importance(var,model,name):
    y_pos = np.arange(len(var))
    importance = model.feature_importances_
    plt.bar(y_pos, importance, align='center')
    plt.xticks(y_pos, var,rotation='vertical')
    plt.ylabel('Feature Importance')
    plt.title('BDT AdaBoost Feature Importance Left Out ' + name)

    plt.savefig('feature_importance_adaboost_'+name+'.pdf',bbox_inches = "tight")
    
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
    var = pd.read_csv(options.variables)
    var = list(var.columns.values)
    X = all_data
    y = all_data['event']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    
    wtrain = X_train['w']
    wtest = X_test['w']
        
    sys.stdout = open('model_out.txt','wt')
    # Training models for different combinations of variables
    fprs = []; tprs = []; aucs = []
    for i in range(0,len(var)):
        new_vars = var[:i]+var[i+1:]
                    
        X_train_new = X_train[new_vars]
        X_test_new = X_test[new_vars]
        model = GradientBoostingClassifier(n_estimators=200,
                                     max_depth=3,
                                     subsample=0.5,
                                     max_features=0.5,
                                     learning_rate=0.1)
        #model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),learning_rate=0.05,n_estimators=400,random_state=0)
        model = model.fit(X_train_new,y_train,sample_weight = wtrain)
        # Getting predictions and plotting roc curve
        predictions = model.predict(X_test_new)
        probs = model.predict_proba(X_test_new)
        fpr,tpr,threshold = metrics.roc_curve(y_test,model.decision_function(X_test_new),sample_weight=wtest)
        auc = metrics.auc(fpr,tpr)
        fprs.append(fpr); tprs.append(tpr); aucs.append(auc)
        compare_train_test(model,X_train_new, y_train, X_test_new, y_test,var[i]) 
        plot_probs(y_test,probs,"gb"+var[i])
        # Printing classification metrics 
        print("Leftout variable")
        print(var[i])
        print("Training input variables:")
        print(new_vars)
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(y_test, predictions,sample_weight=wtest))
        print(metrics.classification_report(y_test, predictions,sample_weight=wtest)) 
        #print(dict(zip(list(X_train.columns.values),model.feature_importances_)))
    # Plotting ROC curves on one graph    
    plt.clf()
    for i in range(0,len(fprs)):
        plt.plot(fprs[i], tprs[i], lw=1, label=var[i]+'(auc = %0.2f)'%(aucs[i]))
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Left Out Variables')
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("roc.pdf",bbox_inches = "tight")
if __name__ == '__main__':
    main()
