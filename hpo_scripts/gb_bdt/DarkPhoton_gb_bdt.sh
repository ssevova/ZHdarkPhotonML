hpogrid model_config recreate DarkPhoton_gb_bdt.sh --script bdtTrain.py --model DarkPhoton_gb_bdt --param '{"run_mode":"local}'
hpogrid search_space recreate bdt_space '{"n_estimators":{"method":"uniform","dimension":{"low":200,"high":900}},"max_depth":{"method":"categorical","dimension":{"categories":[2,3,4,5,6,7,8]}},"sub_sample":{"method":"uniform","dimension":{"low":0.5,"high":0.95"}},"max_features":{"method":"categorical","dimension":{"categories":[1,2,3,4,5,6,7,8,9,10]}},"learning_rate":{"method":"loguniform","dimension":{"low":1e-04,"high":0.1}},"min_samples_split":{"method":"categorical","dimension":{"categories":[2,4,6,8,10,20,40,60,100]}},"min_samples_leaf":{"method":"categorical","dimension":{"categories":[1,3,5,7,9]}}}'

hpogrid hpo_config recreate nevergrad_max_auc --metric auc --mode max --num_trials 1 --algorithm nevergrad --max_concurrent 1
# What is this?
hpogrid grid_config recreate grid_run --inDS user.ssevova:ZHyyD_NN_mc16d_v09
hpogrid project recreate DarkPhoton_gb_bdt --model_config DarkPhoton_gb_bdt --search_space bdt_space --hpo_config nevergrad_max_auc --grid_config grid_run --scripts_path /afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/zhdarkphotonml/python
