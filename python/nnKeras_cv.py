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
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
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
                        default = '/afs/cern.ch/user/e/ehofgard/public/ZHdarkPhoton/source/Plotting/HInvPlot/macros/all_data')
    parser.add_argument('-o',
                        '--output',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')

    return parser
##############################################################################
def create_model(neurons_1 = 160,neurons_2 = 135, neurons_3 = 8, neurons_4 = 8,learn_rate= 0.001,dropout_rate = 0.0, weight_constraint = 0,input_dim=None,batch_size=32,epochs=50,shuffle=True,layers=4,init_mode='lecun_uniform',momentum=0,activation='relu'):
    #,dropout_rate2=0,weight_constraint1=0,weight_constraint2=0):
    # create model
    model = Sequential()
    model.add(Dense(neurons_1,activation=activation,input_dim=input_dim,kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    if layers == 2:
        model.add(Dense(neurons_2, activation=activation,kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
    if layers == 3:
        model.add(Dense(neurons_3, activation=activation,kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
    if layers == 4:
        model.add(Dense(neurons_4,activation=activation,kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid',kernel_initializer=init_mode))
    # Compile model
    optimizer = RMSprop(learning_rate= learn_rate,momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['Recall','AUC','accuracy'])
    return model
def main():
    """ Run script"""
    options = getArgumentParser().parse_args()
    seed = 7
    np.random.seed(seed)
    outfile = 'model_cv_new_100_4layer.txt'
    ### Make output dir
    dir_path = os.getcwd()
    out_dir = options.outdir
    path = os.path.join(dir_path, out_dir)
    if not os.path.exists(path):
        #shutil.rmtree(path)
        os.makedirs(path)
    os.chdir(path)

    all_data = pd.read_csv(options.infile)

    #all_data['w'] = all_data['w'].abs()
    all_data = all_data[all_data['w']>0]

    varw = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']
    units = ['GeV','Radians','GeV','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians']

    #varw = ['met_tight_tst_et','mT','dphi_mety_ll','w']

    # Load the data & split by train/test
    X = all_data
    y = all_data['event']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    X_train = X_train[varw]
    X_test = X_test[varw]
    cols = X_train.columns
    itrain = X_train.index
    itest = X_test.index
    wtest_unscaled = X_test['w']
    # Scaling 
    mapper = DataFrameMapper([(cols,StandardScaler())])
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
    # Building model
    model = KerasClassifier(build_fn=create_model,input_dim=len(varw),verbose=0,neurons_1=12,neurons_2=8,neurons_3=8,neurons_4=8)
    # Different hyperparameters to optimize
    batch_size = [8,16,32,64,128]
    epochs = [50, 100]
    neurons = [5, 10, 15, 20, 25, 30]
    learn_rate = [.00001,.0001,.001,.01,.1]
    #learn_rate = np.logspace(-5,0,25)
    momentum = [0,.5,.9,0.8,.99]
    neurons_1 = np.arange(175,185,1).tolist()
    neurons_2 = np.arange(125,135,1).tolist()
    neurons_3 = np.arange(100,110,1).tolist()
    neurons_4 = np.arange(20,30,1).tolist()
    layers = [2,3,4]
    
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate1 = list(np.arange(0,.9,.05))
    dropout_rate = np.arange(0,.9,.05).tolist()
    weight_constraint = [1,2,3,4,5]
    weight_constraint2 = list(np.arange(1,5,.5))
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # Parameter grid to search
    # For actual, do epochs=100,batch_size=128
    param_grid = dict(neurons_1=neurons_1,neurons_2=neurons_2,neurons_3=neurons_3,learn_rate=learn_rate,neurons_4=neurons_4,dropout_rate=dropout_rate,weight_constraint=weight_constraint,init_mode=init_mode,momentum=momentum,batch_size=batch_size,epochs=epochs)
    #param_grid = dict(dropout_rate1=dropout_rate1,dropout_rate2=dropout_rate2,weight_constraint1=weight_constraint1,weight_constraint2=weight_constraint2)
    # Scoring metrics
    scoring = {'AUC': 'roc_auc', 'Accuracy': metrics.make_scorer(metrics.accuracy_score), 
    #'Recall': metrics.make_scorer(metrics.recall_score),
    'Balanced Accuracy': metrics.make_scorer(metrics.balanced_accuracy_score),'f1': metrics.make_scorer(metrics.f1_score)}
    sys.stdout = open(outfile,'wt')
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
        print(classification_report(y_true, y_pred,sample_weight=wtest_unscaled))
        print(confusion_matrix(y_test, y_pred,sample_weight=wtest_unscaled))
        print()

if __name__ == '__main__':
    main()
           
