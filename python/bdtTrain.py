"""
Script for gradient boosted decision tree hyperparameter tuning.
"""
__author___ = "Stanislava Sevova, Elyssa Hofgard"
###############################################################################
import argparse
# ray tune
from ray import tune

class DarkPhoton_BDT(tune.Trainable):

    def setup(self, config):
        self.n_estimators = int(config.get("n_estimators",200))
        self.max_depth = config.get("max_depth",3)
        self.subsample = config.get("subsample",0.5)
        self.max_features = config.get("max_features",0.5)
        self.learning_rate = config.get("learning_rate",0.1)
        self.min_samples_split = config.get("min_samples_split",2)
        self.min_samples_leaf = config.get("min_samples_leaf",1)
        self.warm_start = config.get("warm_start",False)

        self.varw = config.get("varslist",['met_tight_tst_et'])
        self.run_mode = config.get("run_mode","local")

        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators = self.n_estimators,
                                           max_depth = self.max_depth,
                                           subsample=self.subsample,
                                           max_features=self.max_features,
                                           learning_rate=self.learning_rate,
                                           min_samples_split = self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           warm_start=self.warm_start)
        print(model.get_params())
        self.model = model

    def step(self):                
        import pandas as pd
        import sklearn.model_selection as model_selection
        import sklearn as skl
        import sklearn_pandas as skp
        import hpogrid
        import glob
        
        if self.run_mode == "grid":
            data_directory = hpogrid.get_datadir()
            print("inDS dir: {}".format(data_directory))
            data_files = glob.glob(data_directory+"*.csv")
            print("--> inDS file: {}".format(data_files))
            for f in data_files:
                all_data = pd.read_csv(f)
        elif self.run_mode == "local":
            all_data = pd.read_csv('/afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/zhdarkphotonml/samples/v09/mc16d_v09_samples.csv')

        # Remove negative weights for training
        all_data = all_data[all_data['w']>0]

        # Load the data & split by train/test
        X = all_data
        y = all_data['event'].astype(int)
        X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y,test_size=0.3, random_state=0)
        wtest = X_test['w']
        wtrain = X_train['w']

        self.varw.remove("w")
        X_train = X_train[self.varw]
        X_test = X_test[self.varw]

        history = self.model.fit(X_train, y_train, sample_weight=wtrain)
        # Could also add recall, precision etc
        train_acc = history.score(X_train,y_train) 
        test_acc = history.score(X_test, y_test)
        predictions = history.predict(X_test)        
        probs = history.predict_proba(X_test)
        y_score = history.decision_function(X_test)

        fpr,tpr,threshold = skl.metrics.roc_curve(y_test,y_score,sample_weight=wtest)
        auc = skl.metrics.auc(fpr,tpr) 
        
        return {
            #"epoch": self.iteration,
            #"loss": train_loss,
            "train_accuracy": train_acc,
            "auc": auc,
            "test_accuracy": test_acc
        }
  
           
