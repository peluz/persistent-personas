#!/bin/bash

mamba env create -f environment.yml
mamba activate persistent-personas
python -m ipykernel install --user --name persistent-personas
