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
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
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
                        default = '/afs/cern.ch/user/e/ehofgard/public/ZHdarkPhoton/source/Plotting/HInvPlot/macros/all_data')
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

def plot_dendrogram(X,data_type):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=list(X.columns.values), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.title('Dendrogram and Correlations ' + data_type)
    plt.savefig('dendrogram_'+data_type+'.pdf')

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
    #all_data = all_data[all_data['w']>0]
    #all_data['w'] = all_data['w'].abs()
    # Variables of interest
    var = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']
    units = ['GeV','Radians','GeV','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians','','']
    df_bkg = all_data[all_data['event']==0][var]
    df_sig = all_data[all_data['event']==1][var]

    #for i in range(0,len(var)):
    #    makePlots(df_bkg,df_sig,var[i],units[i],cuts=[])

    # Split into training and testing set and multiply by weights
    X = all_data
    y = all_data['event']
    # Plot all correlations
    #correlations(df_bkg,'Bkg')
    #correlations(df_sig,'Sig')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    
    wtrain = X_train['w']
    wtest = X_test['w']
    X_train = X_train[var]
    X_test = X_test[var]
    plot_dendrogram(df_bkg,'Bkg')
    plot_dendrogram(df_sig,'Sig')
    plot_dendrogram(X[var],'All')

if __name__ == '__main__':
    main()
