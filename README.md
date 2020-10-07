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
2. Create the configurations and project needed to run/submit to grid. Here you will define the name of the project!
```
source hpo_scripts/DarkPhoton_NN.sh
```
3. Run the hyperparameter optimization ...
- locally:
```
hpogrid run <project_name>
```
- on the grid:
```
hpogrid submit <project_name> --n_jobs <number of jobs>
```
4. If you ran on the grid, you can monitor the progress of your jobs by
```
hpogrid tasks show -d <n days since jobs were submitted> -n *.<project_name>.*
```
5. When your grid jobs are done, you can see the results by doing
```
hpogrid report <project_name> -d <n days since jobs were submitted> --to_html
```
There are options to save to html, csv (`--to_csv`) or mlflow (`--to_mlflow`)
