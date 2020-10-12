#!/usr/bin/env python
"""
Train MLP classifer to identify ZH dark photon production against background
"""
__author__ = "Elyssa Hofgard, Stanislava Sevova"
###############################################################################
# Import libraries
##################
import numpy as np
import argparse
import sys
import os
import re
import glob
import shutil
import uproot as up
import uproot_methods
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
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
from sklearn import preprocessing
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
#keras
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
#sklearn_pandas
from sklearn_pandas import DataFrameMapper
# Function to create model, required for KerasClassifier
# Command line arguments
######################## 
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Script for running optimization for the ZH dark photon SR")
    parser.add_argument('-i',
                        '--infile',
                        dest='infile',
                        help='Input CSV file',
                        default = '/afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/zhdarkphotonml/samples/v09/mc16d_v09_samples.csv')
    parser.add_argument('-o',
                        '--output',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')

    return parser
##############################################################################
def create_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(256,activation='relu',input_dim = input_dim,kernel_initializer='glorot_normal'))
    model.add(Dropout(0.813755))
    model.add(Dense(256, activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dropout(0.813755))
    model.add(Dense(64,activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dropout(0.813755))
    model.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_normal'))

    optimizer = RMSprop(learning_rate=0.001720)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy','AUC','Recall'])
    return model

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

def plot_probs(y_test,probs,weight,model):
    prediction = probs
    arr_true_0_indices = (y_test == 0.0)
    arr_true_1_indices = (y_test == 1.0)
    plt.clf()
    arr_pred_0 = prediction[arr_true_0_indices]
    arr_pred_1 = prediction[arr_true_1_indices]
    weight_0 = weight[arr_true_0_indices]
    weight_1 = weight[arr_true_1_indices]
    plt.hist(arr_pred_0, bins=40, label='Background', density=True, histtype='step')
    plt.hist(arr_pred_1, bins=40, label='Signal', density=True, histtype='step')
    plt.xlabel('Probability of being Positive Class')
    plt.legend()
    plt.title('Output Test Probabilities')
    plt.savefig('output_prob_' + model + ".pdf")

def plot_history(history):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_accuracy.pdf')

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_loss.pdf')

    plt.clf()
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_auc.pdf')

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

    #all_data['w'] = all_data['w'].abs()
    all_data = all_data[all_data['w']>0]

    # Variables of interest
    varw = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']
    units = ['GeV','Radians','GeV','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians']
    
    #varw = ['met_tight_tst_et','mT','dphi_mety_ll','w']
    
    # Load the data & split by train/test
    X = all_data
    y = all_data['event']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    seed = 7
    np.random.seed(seed)
    X_train = X_train[varw]
    X_test = X_test[varw]
    cols = X_train.columns
    itrain = X_train.index
    itest = X_test.index
    wtest_unscaled = X_test['w']
    # Scaling 
    mapper = DataFrameMapper([(cols,preprocessing.StandardScaler())])
    scaled_train = mapper.fit_transform(X_train.copy(),len(cols))
    scaled_test = mapper.fit_transform(X_test.copy(),len(cols))
    X_scaled_train = pd.DataFrame(scaled_train,index = itrain, columns=cols)
    X_scaled_test  = pd.DataFrame(scaled_test,index = itest,columns=cols)
    wtest = X_scaled_test['w']
    wtrain = X_scaled_train['w']
    # Deal with weights
    # Build and train the classifier model
    varw.remove("w")
    X_train = X_scaled_train[varw]
    X_test = X_scaled_test[varw]
    model = create_model(len(varw))
    history = model.fit(X_train,y_train,batch_size = 1024,verbose = 0, epochs=100,validation_data=(X_test, y_test))
    #, shuffle=True)#sample_weight=wtrain)
    plot_history(history)
    probs = model.predict(X_test)
    predictions = model.predict_classes(X_test)
    fpr,tpr,threshold = metrics.roc_curve(y_test,probs,sample_weight=wtest_unscaled)
    auc = metrics.auc(fpr,tpr)
    plot_roc(fpr,tpr,auc)
    plot_probs(y_test,probs,wtest_unscaled,"nn")

    sys.stdout = open('model_out.txt','wt')
    '''
    print('Accuracy:')
    print(metrics.accuracy_score(y_test,predictions,sample_weight=wtest))
    print("ROC:")
    print(metrics.roc_auc_score(y_test, probs[:, 1],sample_weight=wtest))
    '''

    # For scikit-learn metrics, use unscaled weights otherwise results make no sense
    print("Training input variables:")
    print(varw)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, predictions,sample_weight=wtest_unscaled))
    print(metrics.classification_report(y_test,predictions,sample_weight=wtest_unscaled))

if __name__ == '__main__':
    main()
           
