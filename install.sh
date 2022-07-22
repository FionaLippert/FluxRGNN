#!/usr/bin/env bash

# install dependencies
echo "Installing environment..."
conda config --set channel_priority strict
conda env create -f environment.yml
source activate fluxrgnn

# install python package
echo "Installing FluxRGNN package"
python setup.py install
