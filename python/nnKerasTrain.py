import argparse

# ray tune
from ray import tune
# Variables of interest
varw = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']

class DarkPhoton_NN(tune.Trainable):

    def _setup(self, config):
        self.n1 = config.get("n1", 160)
        self.n2 = config.get("n2", 135)
        self.n3 = config.get("n3", 8)
        self.n4 = config.get("n4", 8)
        self.dr = config.get("dropout_rate", 0.0)
        self.init_mode = config.get("init_mode","lecun_uniform")
        self.lr = config.get("lr",1e-4)
        self.batchsize = config.get('batchsize', 64)

        import tensorflow as tf

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.n1, activation="relu",kernel_initializer=self.init_mode, input_shape=(len(varw)-1,)))
        model.add(tf.keras.layers.Dropout(self.dr))
        model.add(tf.keras.layers.Dense(self.n2, activation="relu",kernel_initializer=self.init_mode))
        model.add(tf.keras.layers.Dropout(self.dr))
        model.add(tf.keras.layers.Dense(self.n3, activation="relu",kernel_initializer=self.init_mode))
        model.add(tf.keras.layers.Dropout(self.dr))        
        model.add(tf.keras.layers.Dense(self.n4, activation="relu",kernel_initializer=self.init_mode))
        model.add(tf.keras.layers.Dropout(self.dr))        
        model.add(tf.keras.layers.Dense(1, activation="sigmoid",kernel_initializer=self.init_mode))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = self.lr), 
                      loss = tf.keras.losses.BinaryCrossentropy(),
                      metrics=['AUC'])
        print(model.summary())
        self.model = model

    def _train(self):                
        import pandas as pd
        import sklearn.model_selection as model_selection
        import sklearn as skl
        import sklearn_pandas as skp
        import hpogrid
        import glob

        data_directory = hpogrid.get_datadir()
        print("inDS dir: {}".format(data_directory))
        data_files = glob.glob(data_directory+"*.csv")
        print("--> inDS file: {}".format(data_files))
        for f in data_files:
            all_data = pd.read_csv(f)
        all_data = all_data[all_data['w']>0]

        # Load the data & split by train/test
        X = all_data
        y = all_data['event']
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,test_size=0.3, random_state=0)
        X_train = X_train[varw]
        X_test = X_test[varw]
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
        varw.remove("w")
        X_train = X_scaled_train[varw]
        X_test = X_scaled_test[varw]
        
        history = self.model.fit(X_train, y_train, epochs=50,
                                 batch_size=self.batchsize,
                                 validation_data=(X_test, y_test))
        
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=2)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=2)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss": train_loss,
            "accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        }
  
           
