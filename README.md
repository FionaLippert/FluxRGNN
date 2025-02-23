# FluxRGNN
A spatio-temporal modeling framework for large-scale migration forecasts based on 
static sensor network data (e.g. from weather radars).
FluxRGNN is a recurrent graph neural network that is based on a generic
mechanistic description of population-level movements across space and time.
Unlike previous approaches, this hybrid model capitalises on local associations between environmental conditions and migration 
intensity as well as on spatio-temporal dependencies inherent to the movement process.

The original FluxRGNN approach models movements on the Voronoi tessellation of sensor locations (the paper can be found [here](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14007)).
In our follow-up [paper](https://openreview.net/forum?id=oAmxqO1nRy) we have introduced FluxRGNN+, which extends the original FluxRGNN approach to arbitrary tessellations by decoupling the sensor network from the computational grid on which movements are modeled.
This repository provides the code for both approaches. The original implementation of the FluxRGNN approach and the associated experiments and analysis scripts can be found under version [v1.1.1](https://github.com/FionaLippert/FluxRGNN/releases/tag/v.1.1.1). Note that the latest version may not be compatible with all settings and experiments of the original paper.



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
get familiar with the structure of config files. The default settings correspond to the settings used in our FluxRGNN+ paper.

You can, for example, easily switch between models (e.g. `FluxRGNN+` and `FluxRGNN_voronoi`), by simply adding `model=FluxRGNN+` or
`datasource=FluxRGNN_voronoi` to your command line when running one of the provided scripts. Similarly, you could change 
the forecasting horizon to, say, 24 hours by adding `model.horizon=24`.

### Dataloader
The dataloader expects the preprocessed data (including environmental and sensor network data) 
to be in the following path:
``` 
FluxRGNN/data/preprocessed/{t_unit}_{edge_type}_{buffer}_{info}/{datasource}/{season}/{year}
```
where `info` can be either `ndummy={n_dummy_radars}` (if `edge_type` is set to `voronoi`) or `res={h3_resolution}` 
(if `edge_type` is set to `hexagons`).
The values of `t_unit`, `edge_type`, `buffer`, `n_dummy_radars`, `h3_resolution`, `datasource`, `season` and `year` 
can be specified in the hydra configuration files in the `scripts/conf` directory.

To run the preprocessing of weather radar data and atmospheric reanalysis data, 
you can use the script `scripts/run_preprocessing.py` in combination with [this](https://github.com/FionaLippert/birdMigration) code base.
Alternatively, preprocessed European data (`datasource=radar`) can be downloaded [here](https://zenodo.org/records/6874789). 

The preprocessed data must include:
- `delaunay.graphml`: graph structure of the tessellation
- `tessellation.shp`: shape file specifying the geometry of grid cells
- `radar_buffers.shp`: shape file specifying the geometry of radar buffers (i.e. the area around the radar antenna used to obtain measurements)
- `static_cell_features.csv`: dataframe containing static features of cells, e.g. coordinates and cell areas
- `static_radar_features.csv`: dataframe containing static features of radars, e.g. coordinates and antenna altitude
- `dynamic_cell_features.csv`: dataframe containing dynamic features of cells, e.g. temperature and humidity per time point
- `dynamic_radar_features.csv`: dataframe containing dynamic features of radars, e.g. bird densities per time point
- `cell_to_radar_edges.csv`: dataframe specifying the information flow from grid cells to radars (observation model)
- `radar_to_cell_edges.csv`: dataframe specifying the information flow from radars to grid cells (for forecast initialization)

### Training and testing

To train FluxRGNN+ on NEXRAD data, switch to the `scripts` directory and run
```
python run_neural_nets.py model=FluxRGNN+ datasource=nexrad model.scale=0.001 season=fall
```
for fall migratory movements, or 
```
python run_neural_nets.py model=FluxRGNN+ datasource=nexrad model.scale=0.002 season=spring
```
for spring migratory movements.

To run the same on a cluster using slurm and cuda, training for 300 instead of the default 500 epochs, run
```
sbatch run_neural_nets.job 'model=FluxRGNN+ datasource=nexrad model.scale={scale} season={season} device=cluster trainer.max_epochs=300'
```

To generate predictions using a trained model which is stored in `/path/to/model.ckpt`, run
```
python run_neural_nets.py model=FluxRGNN+ datasource=nexrad model.scale={scale} season={season} task=predict model.load_states_from=/path/to/model.ckpt model.horizon={horizon}
```
where `horizon` can be freely adjusted depending on how far into the future you would like to forecast.

The years used for training, validation, and final predictions can be adjusted in the `datasource` config file.

### Contrastive explanations

To analyse the short-term effects of weather on predicted migratory movements using Shapley value-based contrastive explanations, first download the adjusted version of the SHAP python package from [this](https://github.com/FionaLippert/shap) repository, and make sure that it is available in the Python path.

Then, you can run
```
python explain_forecast.py model=FluxRGNN+ datasource=nexrad task=explain model.load_states_from={/path/to/model.ckpt} model.horizon={horizon} model.scale=0.001 task.n_seq_samples=100 task.seqID_start=31 task.seqID_end=76 season=fall
```
to explain 100 randomly chosen nights during the fall peak migration period (`seqID_start` and `seqID_end` are specific for the provided NEXRAD dataset and correspond to the period between 1 September and 15 October). To do the same for the spring peak migration period (10 April to 25 May), run
```
python explain_forecast.py model=FluxRGNN+ datasource=nexrad task=explain model.load_states_from={/path/to/model.ckpt} model.horizon={horizon} model.scale=0.002 task.n_seq_samples=100 task.seqID_start=41 task.seqID_end=86 season=spring
```
This will estimate Shapley values for a range of model outputs (for FluxRGNN+ this includes bird densities, take-off, landing, migration traffic rate, flight direction, and flight speed).

The [Weights&Biases](https://wandb.ai/home) sweep configuration `scripts/sweep_explanations_spring.yaml` and `scripts/sweep_explanations_fall.yaml` can be used to easily run multiple nights in parallel.