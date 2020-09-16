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
from plotUtils import makeHTML
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
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
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
def get_models_n_est():
   """ Tuning the number of estimators for AdaBoost"""
   models = dict()
   #models['10'] = AdaBoostClassifier(n_estimators=10)
   models['50'] = GradientBoostingClassifier(n_estimators=50)
   models['100'] = GradientBoostingClassifier(n_estimators=100)
   models['200'] = GradientBoostingClassifier(n_estimators=200)
   models['300'] = GradientBoostingClassifier(n_estimators=300)
   models['400'] = GradientBoostingClassifier(n_estimators=400)
   models['500'] = GradientBoostingClassifier(n_estimators=500)
   return models  

def get_models_depth():
   models = dict()
   for i in range(1,10):
       models[str(i)] = GradientBoostingClassifier(max_depth=i)
   return models

def get_models_lr():
   models = dict()
   for i in np.arange(0.05, 1.0, 0.05):
       key = '%.3f' % i
       models[key] = GradientBoostingClassifier(learning_rate=i)
   return models
def evaluate_model(model,X,y):
   cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
   scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
   return scores
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
            
    ax.hist(bkg, bins=50, range=(xmin, xmax), histtype='step', color='Red',label='bkg')
    ax.hist(sig, bins=50, range=(xmin, xmax), histtype='step', color='Blue',label='sig')
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
             histtype='stepfilled', density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
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
    varw = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']
    units = ['GeV','Radians','GeV','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians']
    df_bkg = all_data[all_data['event']==0][var]
    df_sig = all_data[all_data['event']==1][var]
    #for i in range(0,len(var)):
        #makePlots(df_bkg,df_sig,var[i],units[i],cuts=[])

    # Split into training and testing set and multiply by weights
    X = all_data
    y = all_data['event']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    X_train = X_train[varw]
    X_test = X_test[varw]
    wtrain = X_train['w']
    wtest = X_test['w']
    cols = X_train.columns
    itrain = X_train.index
    itest = X_test.index
    mapper = DataFrameMapper([(cols,StandardScaler())])
    scaled_train = mapper.fit_transform(X_train.copy(),len(cols))
    scaled_test = mapper.fit_transform(X_test.copy(),len(cols))
    X_train = pd.DataFrame(scaled_train,index = itrain, columns=cols)
    X_test = pd.DataFrame(scaled_test,index = itest,columns=cols)
    X_train = X_train.drop(['w'],axis=1)
    X_test = X_test.drop(['w'],axis=1)
    model = MLPClassifier(max_iter=2000,activation='relu',alpha=0.06,hidden_layer_sizes = (120,75),learning_rate = 'adaptive',momentum=0.9,solver='sgd',batch_size=50,learning_rate_init=0.05)
    '''
    mlp = MLPClassifier(max_iter=2000,batch_size=50,momentum=0.9)
    param_grid = {
    'hidden_layer_sizes': [(sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),), 
                                          (sp_randint.rvs(100,600,1),)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': uniform(0.0001, 0.9),
    'learning_rate': ['constant','adaptive']}
    #{'alpha': 0.06640793542453478, 'batch_size': 50, 'hidden_layer_sizes': (117, 74), 'learning_rate': 'adaptive', 'learning_rate_init': 0.05421689357774788, 'momentum': 0.9, 'solver': 'sgd'}
    #{'alpha': 0.015786202068122347, 'hidden_layer_sizes': (197,), 'learning_rate': 'adaptive', 'learning_rate_init': 0.010660992530318792, 'solver': 'sgd'}
    parameter_space = {
    'hidden_layer_sizes': [(sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),),(sp_randint.rvs(100,600,1),)],
    #'momentum': [0.9,0.95,0.99],
    'solver': ['sgd', 'adam','lbfgs'],
    'alpha': uniform(0.0001,0.1),
    'learning_rate_init': uniform(0.0001,0.1),
    'learning_rate': ['constant','adaptive','invscaling']
    #'batch_size': [10,50,200]
    }
    scores = ['roc_auc']
    sys.stdout = open('model_cv.txt','wt')
    for score in scores:
        clf = RandomizedSearchCV(mlp,
                               parameter_space,
                               cv=3,
                               scoring=score,
                               n_jobs=-1,n_iter=25)
        clf.fit(X_train, y_train)
        print(score)
        print()
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    '''    
    model = model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    metrics.plot_roc_curve(model, X_test, y_test,sample_weight=wtest)
    plt.savefig("roc.pdf")
    metrics.plot_precision_recall_curve(model,X_test,y_test,sample_weight=wtest)
    plt.savefig("prec_recall.pdf")
    #compare_train_test(model,X_train, y_train, X_test, y_test) 
    plot_probs(y_test,probs,"nn")
    # Show output BDT score plot
    '''
    fig,ax = plt.subplots(1,1)
    twoclass_output = model.decision_function(X_test)
    train_output = model.decision_function(X_train)
    class_names = ["Signal", "Background"]
    plot_colors = ['red', 'blue']
    for i, n, c in zip(range(2), class_names, plot_colors):
        ax.hist(twoclass_output[i],
             bins=50,
             range=[-5,5],
             facecolor=c,
             label='Test %s' %n,
             alpha=.5,
             edgecolor=c)
        ax.hist(train_output[i],
            bins=50,
            range=[-5,5],
            label='Train %s' %n,
            fill=False,
            linestyle='--',
            edgecolor=c)
        ax.legend(loc='upper right')
        ax.set_ylabel('Samples')
        ax.set_xlabel('Score')
        ax.set_title('Decision Scores')
        plt.savefig('bdt_train_test_output_scores.pdf') 
    '''

    sys.stdout = open('model_out.txt','wt')
    print('Accuracy:')
    print(metrics.accuracy_score(y_test,predictions,sample_weight=wtest))
    print("ROC:")
    print(metrics.roc_auc_score(y_test, probs[:, 1],sample_weight=wtest))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, predictions,sample_weight=wtest))
    print(metrics.classification_report(y_test, predictions,sample_weight=wtest))
    
if __name__ == '__main__':
    main()
