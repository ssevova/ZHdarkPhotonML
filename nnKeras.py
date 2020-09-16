import numpy
import argparse
import sys
import os
import re
import glob
import shutil
import uproot as up
import uproot_methods
import pandas as pd
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
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
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
                        default='/afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/plotting/source/Plotting/bkgLists/all_data')
    parser.add_argument('-o',
                        '--output',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')

    return parser
##############################################################################
def create_model(optimizer='adam',neurons=20,learn_rate= 0.001,dropout_rate = 0.0, weight_constraint = 0):
    # create model
    model = Sequential()
    model.add(Dense(neurons,activation='relu',kernel_constraint=maxnorm(weight_constraint)))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    #optimizer = Adam(lr = learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy','AUC'])
    return model
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
    var = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph']
    varw = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']
    units = ['GeV','Radians','GeV','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians']
    X = all_data
    y = all_data['event']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    X_train = X_train[varw]
    X_test = X_test[varw]
    cols = X_train.columns
    itrain = X_train.index
    itest = X_test.index
    mapper = DataFrameMapper([(cols,StandardScaler())])
    scaled_train = mapper.fit_transform(X_train.copy(),len(cols))
    scaled_test = mapper.fit_transform(X_test.copy(),len(cols))
    X_train = pd.DataFrame(scaled_train,index = itrain, columns=cols)
    X_test = pd.DataFrame(scaled_test,index = itest,columns=cols)
    wtrain = X_train['w']
    wtest = X_test['w']
    X_train = X_train.drop(['w'],axis=1)
    X_test = X_test.drop(['w'],axis=1)
    model = KerasClassifier(build_fn=create_model,verbose=0,batch_size=5,epochs=50)
    batch_size = [5,10,25, 50]
    epochs = [10, 50, 100]
    neurons = [5, 10, 15, 20, 25, 30]
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_nor    mal', 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    #,dropout_rate=dropout_rate,weight_constraint=weight_constraint)
    grid = RandomizedSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    '''
    bkg_w = all_data[(all_data['w']< 0) & (all_data['event']==0)]
    bkg_w['w'] = bkg_w['w'].abs()
    bkg_w['event'] = list(np.full(len(bkg_w),1))

    sig_w = all_data[(all_data['w']< 0) & (all_data['event']==1)]
    sig_w['w'] = sig_w['w'].abs()
    sig_w['event'] = list(np.full(len(sig_w),0))

    new_data = all_data[all_data['w'] >= 0]
    print(len(all_data))
    all_data = pd.concat([bkg_w,sig_w,new_data])
    print(len(all_data))
    '''
    grid_result = grid.fit(X_train, y_train,sample_weight=wtrain)
    # summarize results
    print("Best: %f using %s on development set" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param)) 
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, grid.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

if __name__ == '__main__':
    main()
           
