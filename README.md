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


## Getting started

### Hydra config
FluxRGNN makes use of [hydra](https://hydra.cc/) to create a hierarchical configuration which can be composed 
dynamically and allows for overrides through the command line. Have a look at the `scripts/config` folder to 
get familiar with the structure of config files. The default settings correspond to the settings used in our 
paper.

You can, for example, easily switch between data sets (e.g. `radar` and `abm`), by simply adding `datasource=radar` or
`datasource=abm` to your command line when running one of the provided scripts. Similarly, you could change 
the forecasting horizon to, say, 24 hours by adding `model.horizon=24`.

### Dataloader
The FluxRGNN dataloader expects the preprocessed data (including environmental and sensor network data) 
to be in the following path:
``` 
FluxRGNN/data/preprocessed/{t_unit}_{edge_type}_{buffer}_{info}/{datasource}/{season}/{year}
```
where `info` can be either `ndummy={n_dummy_radars}` (if `edge_type` is set to `voronoi`) or `res={h3_resolution}` 
(if `edge_type` is set to `hexagons`).
The values of `t_unit`, `edge_type`, `buffer`, `n_dummy_radars`, `h3_resolution`, `datasource`, `season` and `year` 
can be specified in the hydra configuration files in the `scripts/conf` directory.

To run the preprocessing of weather radar data and atmospheric reanalysis data, 
you can use [this](https://github.com/FionaLippert/birdMigration) code base.

The preprocessed data must include:
- `delaunay.graphml`: graph structure of the tessellation
- `static_cell_features.csv`: dataframe containing static features of cells, e.g. coordinates and cell areas
- `static_radar_features.csv`: dataframe containing static features of radars, e.g. coordinates and antenna altitude
- `dynamic_cell_features.csv`: dataframe containing dynamic features of cells, e.g. temperature and humidity per time point
- `dynamic_radar_features.csv`: dataframe containing dynamic features of radars, e.g. bird densities per time point

### Training and testing

To train FluxRGNN on all available data except for year 2017 and to immediately test it on the held-out data, 
switch to the `scripts` directory and run
```
python run_neural_nets.py datasource={datasource} device=local +experiment={name}
```
with `datasource` being either `radar`, `nexrad` or `abm`, and `name` being any identifier you would like to give 
your experiment.

To run the same on a cluster using slurm and cuda, training for 200 epochs with batch_size 32, run
```
sbatch run_neural_nets.job 'datasource={datasource} +experiment={name} device=cluster trainer.max_epochs=200 dataloader.batch_size=32'
```

To make predictions using a trained model, which has been logged with [wandb](https://wandb.ai), run
```
python run_neural_nets.py datasource={datasource} +experiment={name} device=local task=predict model.load_states_from={wandb_model_artifact}
```