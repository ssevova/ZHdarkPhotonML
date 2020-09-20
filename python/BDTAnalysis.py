#!/usr/bin/env python
"""
Beginning script for boosted decision tree analysis ML signal background applications
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
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
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
    
    return parser
###############################################################################                                   
def makePlots(df_bkg,df_sig,var,units,cuts):
    ### Plot some of the distributions for the signal and bkg
    fig,ax = plt.subplots(1,1)
    bkg = np.array(df_bkg[var])
    sig = np.array(df_sig[var])
    #max_val = max(np.concatenate([bkg,sig]))
    # scaling based on signal data
    max_val = max(sig)
    #print(var,min(sig),max_val)
    if var == 'mll':
        xmin = 60
        xmax = 120
    else: 
        xmin = 0
        xmax = max_val
    
    if cuts is not None:
        for cut in cuts:
            if var == cut[0].split()[0]: 
                is_lowerbound = ">" in cut[0]
                for v in cut[1]:
                    line = ax.axvline(v,color='black',linestyle='--')
                    ax.legend([line], ["lower bound" if is_lowerbound else "upper bound"])
    if var == 'w':        
        xmin = min(min(sig),min(bkg))
        ax.hist(bkg,bins=50, range=(xmin, xmax), histtype='step', color='Red',label='bkg')
        ax.hist(sig, bins=50, range=(xmin, xmax), histtype='step', color='Blue',label='sig')
    else:
        ax.hist(bkg,weights = np.abs(df_bkg['w'].to_numpy()),bins=50, range=(xmin, xmax), histtype='step', color='Red',label='bkg')
        ax.hist(sig,weights = np.abs(df_sig['w'].to_numpy()),bins=50, range=(xmin, xmax), histtype='step', color='Blue',label='sig')
    ax.set_xlabel(var+' [' + units + ']')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig("w_hist_" + var+ ".pdf",format="pdf")


def correlations(data,data_type, **kwds):
    """Calculate pairwise correlation between features.
    
    Extra arguments are passed on to DataFrame.corr()
    """
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))
    
    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    ax1.set_title("Correlations "+data_type)

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)
        
    plt.tight_layout()
    plt.savefig("Correlations_"+data_type+".pdf")

def plot_ada(clf,X_train,y_train,X_test,y_test,cutoff,wtrain,wtest):
    ''' Exploring the second peak with adaBoost results
    Filters data for plotting with some cutoff decision score '''
    plt.clf()
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
    X_train['event']=y_train
    X_test['event'] = y_test
    X_train['w']=wtrain
    X_test['w']=wtest
    train_sig = X_train[X_train['event']==1]
    train_sig['score'] = decisions[0]
    train_bkg = X_train[X_train['event']==0]
    train_bkg['score'] = decisions[1]
    test_sig = X_test[X_test['event']==1]
    test_sig['score'] = decisions[2]
    test_bkg = X_test[X_test['event']==0]
    test_bkg['score'] = decisions[3]

    all_sig = pd.concat([train_sig,test_sig])
    all_bkg = pd.concat([train_bkg,test_bkg])

    all_sig = all_sig[all_sig['score'] <= cutoff]
    all_bkg = all_bkg[all_bkg['score'] <= cutoff]
    return all_sig,all_bkg

  
def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
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
    plt.title("Decision Scores")
    plt.xlabel("Score")
    plt.legend(loc='best')
    plt.savefig("BDT_decision.pdf")

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

def cluster_features(X):
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, .8, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    return selected_features

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
    plt.title('Output Test Probabilities')
    plt.savefig('output_prob_' + model + ".pdf")


def plot_importance(var,model,X,y):
    var = np.asarray(var)
    result = permutation_importance(model,X,y,n_repeats=10,random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(model.feature_importances_)
    tree_indices = np.arange(0, len(model.feature_importances_))+0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices,model.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticklabels(var[tree_importance_sorted_idx])
    ax1.set_yticks(tree_indices)
    ax1.set_ylim(0,len(model.feature_importances_))

    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,labels=var[perm_sorted_idx])
    fig.tight_layout()
    plt.savefig('feature_importance_ada.pdf')
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
    #var = ['mllg','metsig_tst','dphi_met_ph','Ptllg']
    units = ['GeV','Radians','GeV','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians','']
    df_bkg = all_data[all_data['event']==0][var]
    df_sig = all_data[all_data['event']==1][var]
    #for i in range(0,len(var)):
    #    makePlots(df_bkg,df_sig,var[i],units[i],cuts=[])

    # Split into training and testing set and multiply by weights
    X = all_data
    y = all_data['event']
    # Plot all correlations
    correlations(df_bkg,'Bkg')
    correlations(df_sig,'Sig')
    #fig,ax = plt.subplots(1,1)
    #ax.hist(all_data['w'].to_numpy(),bins=50)
    #plt.savefig("weights.pdf")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    
    wtrain = X_train['w']
    wtest = X_test['w']

    # Using hierarchical clustering to see which variables are most important and training with those
    '''
    clustered=cluster_features(all_data[var])
    var = np.asarray(var)
    var = list(var[clustered])
    '''
    X_train = X_train[var]
    X_test=X_test[var]
    model = GradientBoostingClassifier(n_estimators=200,
                                 max_depth=3,
                                 subsample=0.5,
                                 max_features=0.5,
                                 learning_rate=0.1)
    #model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),learning_rate=0.05,n_estimators=400,random_state=0)
    model = model.fit(X_train,y_train,sample_weight = wtrain)
    # Getting predictions and plotting roc curve
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    fpr,tpr,threshold = metrics.roc_curve(y_test,model.decision_function(X_test),sample_weight=wtest)
    auc = metrics.auc(fpr,tpr)
    plot_roc(fpr,tpr,auc)
    #metrics.plot_precision_recall_curve(model,X_test,y_test,sample_weight=wtest)
    #plt.savefig("prec_recall.pdf")
    compare_train_test(model,X_train, y_train, X_test, y_test) 
    plot_probs(y_test,probs,"gb")
    plot_importance(var,model,X_train,y_train)
    # Plotting weird events with second peak adaboost
    '''
    cutoff = -.25
    df_sig_cutoff,df_bkg_cutoff = plot_ada(model,X_train,y_train,X_test,y_test,cutoff,wtrain,wtest)
    for i in range(0,len(var)):
        makePlots(df_bkg_cutoff,df_sig_cutoff,var[i],units[i],cuts=[])
    '''
    # Printing classification metrics 
    sys.stdout = open('model_out.txt','wt')
    print("Training input variables:")
    print(var)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, predictions,sample_weight=wtest))
    print(metrics.classification_report(y_test, predictions,sample_weight=wtest)) 
    print("Feature Importance:")
    print(model.feature_importances_)
    #print(dict(zip(list(X_train.columns.values),model.feature_importances_)))

    
if __name__ == '__main__':
    main()
