[![DOI](https://zenodo.org/badge/450534842.svg)](https://zenodo.org/badge/latestdoi/450534842)

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


Please note that a refactored and extended version of FluxRGNN is available on the branch [nexrad_data](https://github.com/FionaLippert/FluxRGNN/tree/nexrad_data), which will be merged soon into the main code base.

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

Note that after making changes to files in the `fluxrgnn` directory, you need to reinstall the associated python package by running
```
python setup.py install
```

### Additional dependencies

If you want to use your GPU, you may need to manually install a matching 
[PyTorch](https://pytorch.org/) version.

To recreate geographical visualisations from our paper, some additional packages are required. They can be installed by running
```
conda env update --name fluxrgnn --file plotting_environment.yml
```
To make the conda environment visible for the jupyter notebooks, run
```
python -m ipykernel install --user --name=fluxrgnn
```

To install additional packages required to run the radar data preprocessing (see below), run
```
conda env update --name fluxrgnn --file preprocessing_environment.yml
```

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

To reproduce the results from our paper, please download the preprocessed data [here](https://doi.org/10.5281/zenodo.6364940)

To run the preprocessing of bird density and velocity data from 
the European weather radar network yourself, you can use [this](https://github.com/FionaLippert/birdMigration) code base. Follow the README to install the `birds` python package in your `fluxrgnn` conda environment and download the raw radar data. Then, from the `FluxRGNN/scripts` directory, run
```
python run_preprocessing.py datasource=radar +raw_data_dir={path/to/downloaded/data}
```

If you would like to apply FluxRGNN to your own data, you need to generate the following files (for each season and year):
- `delaunay.gpickle`: graph structure underlying the desired tessellation as a [networkx.DiGraph](https://networkx.org/documentation/stable/reference/classes/digraph.html) where nodes represent grid cells and edges between cells exist if they are adjacent. You can use [this](https://github.com/FionaLippert/birdMigration) code base to construct Voronoi or hexagonal tessellations and the associated graph structure from a set of sensor locations.
- `tessellation.shp`: shape file specifying the geometry of grid cells
- `radar_buffers.shp`: shape file specifying the geometry of radar buffers (i.e. the area around the radar antenna used to obtain measurements)
- `cell_to_radar_edges.csv`: dataframe containing edges between grid cells (cidx) and sensors (ridx), the distance between cell centers and sensor locations, and the area of overlap (intersection) between the cell and the sensor measurement area. This graph is used to define a simplified observation model mapping cell quantities to sensor measurements. 
- `radar_to_cell_edges.csv`: dataframe containing edges between sensors (ridx) and grid cells (cidx) and their distance. This graph is used to infer cell quantities from sparse sensor measurements during forecast initialization.
- `static_radar_features.csv`: dataframe containing the following static features of radars:
    |          	| description                                                                      	| data type 	|
    |----------	|----------------------------------------------------------------------------------	|-----------	|
    | ID    	| radar identifier used to define graph structures                                        | integer    	|
    | radar    	| name/label of radar                                           	                | string    	|
    | observed 	| true if data is available for this radar, false otherwise                        	| boolean   	|
    | x        	| x-component of radar location in local coordinate reference system               	| float     	|
    | y        	| y-component of radar location in local coordinate reference system               	| float     	|
    | lon      	| longitude of radar location                                                      	| float     	|
    | lat      	| latitude of radar location                                                       	| float     	|
    | area_km2 	| measurement area in km^2                                                      	| float     	|

- `static_cell_features.csv`: dataframe containing the following static features of grid cells:
    |          	| description                                                                      	| data type 	|
    |----------	|----------------------------------------------------------------------------------	|-----------	|
    | ID    	| cell identifier used to define graph structures                                        | integer    	|
    | h3_id    	| H3 cell identifier (if hexagonal H3 tessellation is used)                      | string    	|
    | radar    	| list of radars located within the cell                                            | List of strings|
    | observed 	| true if at least one radar is located within the cell, false otherwise            | boolean   	|
    | x        	| x-component of cell center location in local coordinate reference system         	| float     	|
    | y        	| y-component of cell center location in local coordinate reference system        	| float     	|
    | lon      	| longitude of cell center location                                                	| float     	|
    | lat      	| latitude of cell center location                                                	| float     	|
    | area_km2 	| cell area in km^2                                                             	| float     	|
    | boundary 	| true if cell is at the boundary of the modeled domain, false otherwise           	| boolean     	|
    | nlcd_maj 	| the NLCD land cover class dominating the cell                                 	| integer     	|
    | nlcd_cX 	| the fraction of the cell covered by NLCD land cover class X (for X=0,...,18)      | float     	|


- `dynamic_radar_features.csv`: dataframe containing the following dynamic radar features, i.e. variables that change over time:
    |                	| description                                                                                                                                                          	| data type 	|
    |----------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-----------	|
    | ID    	        | radar identifier used to define graph structures                                        | integer    	|
    | radar          	| name/label of radar                                                                                                                                	| string    	|
    | datetime       	| timestamp defining the beginning of the time step (e.g. "2015-08-01 12:00:00+00:00")                                                                                 	| string    	|
    | dayofyear      	| day of the year (determined based on the beginning of the time step)                                                                                                 	| int       	|
    | tidx           	| time index used for indexing, sorting and aligning data sequences of multiple radars                                                                                 	| int       	|
    | solarpos | solar position (in degrees) | float |
    | solarpos_dt | change in solar position relative to the previous time step (in degrees) | float |
    | night          	| true if at any point during the time step the sun angle is below -6 degrees, false otherwise                                                                         	| boolean   	|
    | birds_km2      	| bird density (birds/km^2) measured by the radar                                                                                                  	| float     	|
    | bird_u         	| u-component of the bird velocity measured by the radar                                                                                                               	| float     	|
    | bird_v         	| v-component of the bird velocity measured by the radar                                                                                                               	| float     	|
    | missing_birds_km2        	| true if bird density data is missing, false otherwise                                                                                                                             	| boolean   	|
    | missing_bird_uv        	| true if bird_u or bird_v data is missing, false otherwise                                                                                                                             	| boolean   	|
    
    
- `dynamic_cell_features.csv`: dataframe containing the following dynamic cell features, i.e. variables that change over time:
    |                	| description                                                                                                                                                          	| data type 	|
    |----------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-----------	|
    | ID    	        | cell identifier used to define graph structures                                        | integer    	|
    | datetime       	| timestamp defining the beginning of the time step (e.g. "2015-08-01 12:00:00+00:00")                                                                                 	| string    	|
    | dayofyear      	| day of the year (determined based on the beginning of the time step)                                                                                                 	| int       	|
    | tidx           	| time index used for indexing, sorting and aligning data sequences of multiple radars                                                                                 	| int       	|
    | solarpos | solar position (in degrees) | float |
    | solarpos_dt | change in solar position relative to the previous time step (in degrees) | float |
    | dusk           	| true if at any point during the time step the sun angle drops below 6 degrees, false otherwise                                                                       	| boolean   	|
    | dawn           	| true if at any point during the time step the sun angle rises above 6 degrees, false otherwise                                                                       	| boolean   	|
    | night          	| true if at any point during the time step the sun angle is below -6 degrees, false otherwise                                                                         	| boolean   	|
    | nightID | night identifier used to group data belonging to the same night | integer |
    | ...            	| any relevant environmental variables can be added here. The variable names should correspond to those  specified in the env_vars list in the datasource config file. 	|           	|

    


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

To train and evaluate one of the baseline models (`model = HA, GAM, or GBT`), simply add `model={model}` to your command line.



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
