hpogrid model_config recreate DarkPhoton_NN_layer --script nnKerasTrain.py --model DarkPhoton_NN --param '{"run_mode":"grid", "epochs":50}'
hpogrid search_space recreate layer_nn_space '{"dense_layers":{"method":"categorical","dimension":{"categories":[1,2,3,4,5,6]}}, "top_layer_neurons":{"method":"categorical","dimension":{"categories":[512,256,128]}}, "bot_layer_neurons":{"method":"categorical","dimension":{"categories":[128,64,32,16]}}, "dense_activation":{"method":"categorical","dimension":{"categories":["relu","softmax","softplus","tanh","softsign","selu","elu"]}}, "beta_1":{"method":"uniform","dimension":{"low":0.5,"high":1}}, "batchsize":{"method":"categorical","dimension":{"categories":[32,64,128,256,512]}}, "lr":{"method":"loguniform","dimension":{"low":1e-05,"high":0.1}}, "dropout_rate":{"method":"uniform","dimension":{"low":0.5,"high":1}}, "init_mode":{"method":"categorical","dimension":{"categories":["lecun_uniform","uniform","normal", "zero","glorot_normal","glorot_uniform","he_normal","he_uniform"]}}}'
hpogrid hpo_config recreate nevergrad_max_auc --metric auc --mode max --num_trials 1 --algorithm nevergrad
hpogrid grid_config recreate grid_run --inDS user.ssevova:ZHyyD_NN_mc16d_v08 
hpogrid project recreate DarkPhoton_NN_layer --model_config DarkPhoton_NN_layer --search_space layer_nn_space --hpo_config nevergrad_max_auc --grid_config grid_run --scripts_path /afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/zhdarkphotonml/python