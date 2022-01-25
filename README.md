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

### Dataloader
The FluxRGNN dataloader expects the preprocessed data (including environmental and sensor network data) 
to be in the following path:
``` 
FluxRGNN/data/preprocessed/{t_unit}_voronoi_ndummy={ndummy}/{datasource}/{season}/{year}
```
where `t_unit`, `ndummy`, `datasource`, `season` and `year` can be specified in the configuration files 
in the `scripts/conf` directory.

To reproduce results from our paper, please download the preprocessed data here (link to zenodo)
To run the preprocessing of bird density and velocity data from 
the European weather radar network yourself, you can use this code (link to birdMigration repository).

The preprocessed data must include:
- `delaunay.gpickle`: graph structure underlying the Voronoi tessellation of sensor locations
- `static_features.csv`: dataframe containing static features of sensors and their corresponding Voronoi cell, e.g. coordinates, cell areas
- `dynamic_features.csv`: dataframe containing dynamic features of Voronoi cells, e.g. animal densities and wind speed per time point

### Training and testing

### Analysis

## How to cite