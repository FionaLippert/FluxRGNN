#!/usr/bin/env bash

# install dependencies
echo "Installing environment..."
conda env create -f environment.yml
conda activate fluxrgnn

# install python package
echo "Installing FluxRGNN package"
python setup.py install
