import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
import argparse
import sys
import os
import re
import glob
import shutil
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
    plt.savefig("BDT_decision_"+clf+".pdf")

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

def plot_probabilities(probas,y_pred,X_test,name):
    plt.clf()
    n_classes = np.unique(y_pred).size
    X = X_test[:,0:2]
    plt.figure(figsize=(1, n_classes))
    plt.subplots_adjust(bottom=.2, top=.95)
    for k in range(n_classes):
        plt.subplot(1,k+1,k+1)
        plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel(name)
        new_prob = np.reshape(probas[:,k],(len(probas[:,k]),-1))
        imshow_handle = plt.imshow(new_prob,
                                   extent=(3, 9, 1, 5), origin='lower')
        plt.xticks(())
        plt.yticks(())
        idx = (y_pred == k)
        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal') 
    plt.savefig('probs'+name+".pdf")

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
    all_data['w'] = all_data['w'].abs()

    X = all_data
    y = all_data['event']
    var = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    wtrain = X_train['w']
    wtest = X_test['w']
    X_train = X_train[var]
    X_test = X_test[var]

    names = ["Nearest Neighbors",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Gradient Boost","Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),learning_rate=0.1,n_estimators=1000),
        GradientBoostingClassifier(n_estimators=200,
                                 max_depth=3,
                                 subsample=0.5,
                                 max_features=0.5,
                                 learning_rate=0.1),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    sys.stdout = open('model_out.txt','wt')
    for name, model in zip(names,classifiers):
        model = model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics.plot_roc_curve(model, X_test, y_test,sample_weight=wtest)
        plt.savefig("roc_" + name + ".pdf")
        metrics.plot_precision_recall_curve(model,X_test,y_test,sample_weight=wtest)
        plt.savefig("prec_recall_"+name+".pdf")
        plot_probs(y_test,probs,name)
        print("---------------------------------------------------")
        print("Model: " + name)
        print('Accuracy:')
        print(metrics.accuracy_score(y_test,predictions,sample_weight=wtest))
        print("ROC:")
        print(metrics.roc_auc_score(y_test, probs[:, 1],sample_weight=wtest))
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(y_test, predictions,sample_weight=wtest))
        print(metrics.classification_report(y_test, predictions,sample_weight=wtest))

if __name__ == '__main__':
    main()
