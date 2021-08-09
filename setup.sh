#!/bin/bash

WORK_DIR=$(pwd)

# cython is required by some packages in requirements.txt
pip install cython
pip install -r requirements.txt

# install the latest madmom from source code
git clone https://github.com/CPJKU/madmom.git ~/madmom
cd ~/madmom
git submodule update --init --remote
python setup.py develop --user
cd $WORK_DIR
