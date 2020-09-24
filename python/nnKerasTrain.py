import argparse
# ray tune
from ray import tune

class DarkPhoton_NN(tune.Trainable):

    def setup(self, config):
        self.epochs = config.get("epochs",50)
        self.dense_layers = config.get("dense_layers",2)
        self.top_layer_neurons = config.get("top_layer_neurons",256)
        self.bot_layer_neurons = config.get("bot_layer_neurons",16)
        self.dense_activation = config.get("dense_activation","relu")
        '''
        self.n1 = config.get("n1", 160)
        self.n2 = config.get("n2", 135)
        self.n3 = config.get("n3", 8)
        self.n4 = config.get("n4", 8)
        '''
        self.dr = config.get("dropout_rate", 0.0)
        self.init_mode = config.get("init_mode","lecun_uniform")
        self.lr = config.get("lr",1e-4)
        self.batchsize = config.get('batchsize', 64)
        self.run_mode = config.get("run_mode","local")
        import tensorflow as tf

        self.varw = config.get("varslist",['met_tight_tst_et'])
        #['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']

        model = tf.keras.models.Sequential()
        for layer in range(self.dense_layers):
            if layer == 0: 
                model.add(tf.keras.layers.Dense(self.top_layer_neurons, activation=self.dense_activation,kernel_initializer=self.init_mode, input_shape=(len(self.varw)-1,)))
            else: 
                if layer <= self.dense_layers/2:
                    model.add(tf.keras.layers.Dense(self.top_layer_neurons, activation=self.dense_activation, kernel_initializer=self.init_mode))
                else: 
                    model.add(tf.keras.layers.Dense(self.bot_layer_neurons, activation=self.dense_activation, kernel_initializer=self.init_mode))
            model.add(tf.keras.layers.Dropout(self.dr))
            
        model.add(tf.keras.layers.Dense(1, activation="sigmoid",kernel_initializer=self.init_mode))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = self.lr), 
                      loss = tf.keras.losses.BinaryCrossentropy(),
                      metrics=['AUC'])
        print(model.summary())
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
            all_data = pd.read_csv('/afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/plotting/trees/v08/tight-and-ph-skim/mc16d/dataLists/all_data')

        # Remove negative weights for training
        all_data = all_data[all_data['w']>0]

        # Load the data & split by train/test
        X = all_data
        y = all_data['event']
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,test_size=0.3, random_state=0)
        X_train = X_train[self.varw]
        X_test = X_test[self.varw]
        cols = X_train.columns
        itrain = X_train.index
        itest = X_test.index
        wtest_unscaled = X_test['w']

        # Scaling 
        mapper = skp.DataFrameMapper([(cols,skl.preprocessing.StandardScaler())])
        scaled_train = mapper.fit_transform(X_train.copy(),len(cols))
        scaled_test = mapper.fit_transform(X_test.copy(),len(cols))
        X_scaled_train = pd.DataFrame(scaled_train,index = itrain, columns=cols)
        X_scaled_test  = pd.DataFrame(scaled_test,index = itest,columns=cols)
        wtest = X_scaled_test['w']
        wtrain = X_scaled_train['w']

        # Deal with weights
        self.varw.remove("w")
        X_train = X_scaled_train[self.varw]
        X_test = X_scaled_test[self.varw]
        
        history = self.model.fit(X_train, y_train, epochs=self.epochs,
                                 batch_size=self.batchsize,
                                 validation_data=(X_test, y_test))
        
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=2)
        test_loss, test_acc   = self.model.evaluate(X_test, y_test, verbose=2)
        
        probs = self.model.predict(X_test)
        predictions = self.model.predict_classes(X_test)
        fpr,tpr,threshold = skl.metrics.roc_curve(y_test,probs,sample_weight=wtest_unscaled)
        auc = skl.metrics.auc(fpr,tpr) 
        
        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss": train_loss,
            "accuracy": train_acc,
            "auc": auc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        }
  
           
