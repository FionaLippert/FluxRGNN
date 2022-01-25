# FluxRGNN
A spatio-temporal modeling framework for large-scale migration forecasts based on 
static sensor network data.
FluxRGNN is a recurrent graph neural network that is based on a generic
mechanistic description of population-level movements on the Voronoi tessellation of sensor locations. 
Unlike previous approaches, this hybrid model capitalises on local associations between environmental conditions and migration 
intensity as well as on spatio-temporal dependencies inherent to the movement process. 


## Requirements and setup
- python
- conda
- pytorch (if you want to accelerate training with a GPU, make sure to install a cuda-enabled pytorch version)

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

To reproduce the results from our paper, please download the preprocessed data here (link to zenodo)
To run the preprocessing of bird density and velocity data from 
the European weather radar network yourself, you can use this code (link to birdMigration repository).

The preprocessed data must include:
- `delaunay.gpickle`: graph structure underlying the Voronoi tessellation of sensor locations
- `static_features.csv`: dataframe containing static features of sensors and their corresponding Voronoi cell, e.g. coordinates, cell areas
- `dynamic_features.csv`: dataframe containing dynamic features of Voronoi cells, e.g. animal densities and wind speed per time point

### Training and testing

To train FluxRGNN on all available data except for year 2017 and to immediately test it on the held-out data, run
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

## How to cite