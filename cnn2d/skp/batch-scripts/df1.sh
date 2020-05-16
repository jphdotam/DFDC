#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=17g
#SBATCH --out=outfiles/df1.out
#SBATCH -t 144:00:00

source activate pytorch_p37
cd /home/ianpan/ufrc/deepfake/skp/
/home/ianpan/anaconda3/envs/pytorch_p37/bin/python run.py configs/experiments/experiment045.yaml train --gpu 0 --num-workers 4 
