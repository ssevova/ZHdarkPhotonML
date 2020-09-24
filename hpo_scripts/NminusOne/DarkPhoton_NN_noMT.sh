hpogrid model_config create DarkPhoton_NN_noMT --script nnKerasTrain.py --model DarkPhoton_NN --param '{"run_mode":"grid", "epochs":100, "varslist":["met_tight_tst_et","met_tight_tst_phi","ph_pt","dphi_mety_ll","AbsPt","Ptll","mllg","lep1pt","lep2pt","mll","metsig_tst","Ptllg","dphi_met_ph","w"]}'
hpogrid search_space recreate 2layer_nn_space '{"dense_layers":{"method":"categorical","dimension":{"categories":[2]}}, "top_layer_neurons":{"method":"categorical","dimension":{"categories":[512,256]}}, "bot_layer_neurons":{"method":"categorical","dimension":{"categories":[64,32,16]}}, "dense_activation":{"method":"categorical","dimension":{"categories":["relu","softmax","elu"]}}, "beta_1":{"method":"uniform","dimension":{"low":0.6,"high":1}}, "batchsize":{"method":"categorical","dimension":{"categories":[64,128,256,512,1024]}}, "lr":{"method":"loguniform","dimension":{"low":1e-04,"high":0.1}}, "dropout_rate":{"method":"uniform","dimension":{"low":0.5,"high":1}}, "init_mode":{"method":"categorical","dimension":{"categories":["lecun_uniform","uniform","normal","glorot_normal","glorot_uniform"]}}}'
hpogrid hpo_config recreate nevergrad_max_auc --metric auc --mode max --num_trials 100 --algorithm nevergrad --max_concurrent 1 
hpogrid grid_config recreate grid_run --inDS user.ssevova:ZHyyD_NN_mc16d_v08 
hpogrid project create DarkPhoton_NN_noMT --model_config DarkPhoton_NN_noMT --search_space 2layer_nn_space --hpo_config nevergrad_max_auc --grid_config grid_run --scripts_path /afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/zhdarkphotonml/python
