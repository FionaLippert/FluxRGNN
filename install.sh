#!/usr/bin/env bash

# install dependencies
echo "Please make sure conda is installed."
echo "Installing environment..."
conda env create -f environment.yml
conda activate fluxrgnn

# install python package
python setup.py install