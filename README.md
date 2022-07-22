[![DOI](https://zenodo.org/badge/450534842.svg)](https://zenodo.org/badge/latestdoi/450534842)

# FluxRGNN
A spatio-temporal modeling framework for large-scale migration forecasts based on 
static sensor network data.
FluxRGNN is a recurrent graph neural network that is based on a generic
mechanistic description of population-level movements on the Voronoi tessellation of sensor locations. 
Unlike previous approaches, this hybrid model capitalises on local associations between environmental conditions and migration 
intensity as well as on spatio-temporal dependencies inherent to the movement process. 


## Requirements and setup
First, make sure you have [conda](https://docs.conda.io/en/latest/) installed.

To install all other dependencies and the FluxRGNN package itself, switch to the FluxRGNN directory and run:
```
bash install.sh
```
This will create a new conda environment called `fluxrgnn` and will install the FluxRGNN package into this environment.
Later on, it is enough to activate the environment with
```
conda activate fluxrgnn
```
before getting started.

If you want to use your GPU, you may need to manually install a matching 
[PyTorch](https://pytorch.org/) version.

To recreate geographical visualisations from our paper, some additional packages are required. They can be installed by running
```
conda env update --name fluxrgnn --file environment_geo.yml
```


## Getting started

### Hydra config
FluxRGNN makes use of [hydra](https://hydra.cc/) to create a hierarchical configuration which can be composed 
dynamically and allows for overrides through the command line. Have a look at the `scripts/config` folder to 
get familiar with the structure of config files. The default settings correspond to the settings used in our 
paper.

You can, for example, easily switch between data sets (here `radar` and `abm`), by simply adding `datasource=radar` or
`datasource=abm` to your command line when running one of the provided scripts. Similarly, you could change 
the number of fully-connected layers used in FluxRGNN to, say, 3 by adding `model.n_fc_layers=3`.

### Dataloader
The FluxRGNN dataloader expects the preprocessed data (including environmental and sensor network data) 
to be in the following path:
``` 
FluxRGNN/data/preprocessed/{t_unit}_voronoi_ndummy={ndummy}/{datasource}/{season}/{year}
```
where `t_unit`, `ndummy`, `datasource`, `season` and `year` can be specified in the hydra configuration files 
in the `scripts/conf` directory.

To reproduce the results from our paper, please download the preprocessed data [here](https://doi.org/10.5281/zenodo.6364940)
To run the preprocessing of bird density and velocity data from 
the European weather radar network yourself, you can use [this](https://github.com/FionaLippert/birdMigration) code base. Follow the README to install the `birds` python package in your `fluxrgnn` conda environment and download the raw radar data. Then, from the `FluxRGNN/scripts` directory, run
```
python run_preprocessing.py datasource=radar +raw_data_dir={path/to/downloaded/data}
```

The preprocessed data must include:
- `delaunay.gpickle`: graph structure underlying the Voronoi tessellation of sensor locations
- `static_features.csv`: dataframe containing static features of sensors and their corresponding Voronoi cell, e.g. coordinates, cell areas
- `dynamic_features.csv`: dataframe containing dynamic features of Voronoi cells, e.g. animal densities and wind speed per time point

### Training and testing

To train FluxRGNN on all available data except for year 2017 and to immediately test it on the held-out data, switch to the `scripts` directory and run
```
python run_experiments.py datasource={datasource} +experiment={name}
```
with `datasource` being either `radar` or `abm`, and `name` being any identifier you would like to give 
your experiment.

To run the same on a cluster using slurm and cuda, with 5 instances of FluxRGNN being trained in parallel, run
```
python run_experiments.py datasource={datasource} +experiment={name} device=cluster task.repeats=5
```

To train and evaluate one of the baseline models (`model = HA, GAM, or GBT`), simply add `model={model}` to your command line.

### Analysis

#### Predictive performance

To compare the predictive performance of FluxRGNN to the baseline models, run
```
python evaluate_performance.py datasource={datasource} +experiment_type=final
```

Similarly, to compare the predictive performance of FluxRGNN to its variants (ablations), run
```
python evaluate_performance.py datasource={datasource} +experiment_type=ablations
```

This will generate summaries of the performance measures and write them to the directory `FluxRGNN/results/{datasource}/performance_evaluation`.
Then the Jupyter notebook `performance_evaluation.ipynb` can be used to recreate the figures from our paper.

#### Validation of fluxes and source/sink terms

To validate the spatial and temporal component of FluxRGNN by comparing 24h fluxes and source/sink terms to the 
respective ground truth from simulations, run
```
python evaluate_fluxes.py datasource=abm
```

To do the same for hourly fluxes and source/sink terms, run
```
python evaluate_fluxes.py datasource=abm +H_min=24 +H_max=24
```
The forecasting horizon (`H_min` and `H_max`) can be set to anything between 1 and 72.

To recreate the figures from our paper, use the Jupyter notebook `validation_study.ipynb`.

#### Radar case study

To recreate the map of average 24h fluxes predicted for the radar data, first run 
```
python evaluate_fluxes.py datasource=radar
```
and then use the Jupyter notebook `radar_case_study.ipynb` for plotting.

The same notebook can be used to visualize example predictions for a single radar or the entire network.
