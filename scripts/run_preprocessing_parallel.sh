#!/bin/bash

YEAR_MIN=$1
YEAR_MAX=$2
KWARGS=$3


for YEAR in $(seq $YEAR_MIN $YEAR_MAX); do
	sbatch run_preprocessing.job "datasource.years=[${YEAR}] ${KWARGS}"
done
