import numpy as np
import pandas as pd
#keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dropout, RMSprop, Dense
#from tensorflow.keras.constraints import maxnorm
#from tensorflow.keras.optimizers import Adam
#sklearn_pandas

# ray tune
from ray import tune

class CreateModel(Sequential):
    def __init__(self, 
                 n1=160, n2=135, n3=8, n4=8, 
                 init_mode='lecun_uniform',
                 dropout_rate=0.0):
        super(CreateModel, self).__init__()
        self.dr = Dropout(dropout_rate)
        self.l1 = Dense(n1, activation="relu",kernel_initializer=init_mode)
        self.l2 = Dense(n2, activation="relu",kernel_initializer=init_mode)
        self.l3 = Dense(n3, activation="relu",kernel_initializer=init_mode)
        self.l4 = Dense(n4, activation="relu",kernel_initializer=init_mode)
        self.lf = Dense(1, activation="sigmoid",kernel_initializer=init_mode)
        
    def call(self, x):
        x = self.l1(x)
        x = self.dr(x)
        x = self.l2(x)
        x = self.dr(x)
        x = self.l3(x)
        x = self.dr(x)
        x = self.l4(x)
        x = self.dr(x)
        return self.lf(x)
  

class DarkPhoton_NN(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf
        import sklearn as skl
        import sklearn_pandas as skp

        all_data = pd.read_csv("all_data")
        all_data = all_data[all_data['w']>0]

        # Variables of interest
        varw = ['met_tight_tst_et','met_tight_tst_phi','mT','ph_pt','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']
            
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
        
        # Add a channels dimension
        X_train = X_train[..., tf.newaxis]
        X_test = X_test[..., tf.newaxis]


        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.train_ds = self.train_ds.shuffle(10000).batch(
            config.get("batchsize", 32))

        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test,
                                                           y_test)).batch(32) 
        self.model = CreateModel(n1=config.get("n1",160),
                                 n2=config.get("n2",135),
                                 n3=config.get("n3",8),
                                 n4=config.get("n4",8),
                                 dropout_rate=config.get("dropout_rate",0.0),
                                 init_mode=config.get("init_mode","lecun_uniform"))
                             
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get("lr",1e-4),
            beta_1=config.get("beta_1",0.9))
        self.train_loss = kf.keras.metrics.AUC(name="train_loss")
        self.train_accuracy = kf.keras.metrics.BinaryAccuracy(name="train_accuracy")
        self.test_loss = kf.keras.metrics.AUC(name="test_loss")
        self.test_accuracy = kf.keras.metrics.BinaryAccuracy(name="test_accuracy")

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images)
                loss = self.loss_object(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = self.model(images)
            t_loss = self.loss_object(labels, predictions)

            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)

        self.tf_train_step = train_step
        self.tf_test_step = test_step

    def _train(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

        for idx, (images, labels) in enumerate(self.train_ds):
            if idx > MAX_TRAIN_BATCH:  # This is optional and can be removed.
                break
            self.tf_train_step(images, labels)

        for test_images, test_labels in self.test_ds:
            self.tf_test_step(test_images, test_labels)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            "accuracy": self.train_accuracy.result().numpy() * 100,
            "test_loss": self.test_loss.result().numpy(),
            "mean_accuracy": self.test_accuracy.result().numpy() * 100
        }


  
           
