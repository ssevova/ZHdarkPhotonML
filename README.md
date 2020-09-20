# ZHdarkPhotonML

This is a package for hyperparameter optimization of the ZH dark photon NN using ATLAS grid GPU/CPU resources. 

It contains the following directories:
- [hpogrid](https://gitlab.cern.ch/aml/hyperparameter-optimization/alkaid-qt/hpogrid/-/tree/master/): the main work-horse
- `python`: contains the training script to be used
- `hpo_scripts`: contains the script needed to create the configurations to run/submit to the grid

## Quick start

1. Setup the `ml-base` conda environment
```
source hpogrid/setupenv.sh
```
2. Create the configurations and project needed to run/submit to grid
```
source hpo_scripts/DarkPhoton_NN.sh
```
3. Run the hyperparameter optimization ...
- locally:
```
hpogrid run DarkPhoton_NN
```
- on the grid:
```
hpogrid submit DarkPhoton_NN
```