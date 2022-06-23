#!/bin/bash

#$ -N train_the_kitti
#$ -q gpu
#S -m abe
#$ -l gpu=1

BASE_PATH="$HOME/research/pytorch-prednet-mod" 

module load python
source env/bin/activate
# pip3 install -r requirements.txt

python3 kitti_train.py
