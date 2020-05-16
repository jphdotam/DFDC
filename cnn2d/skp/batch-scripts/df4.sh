#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=5
#SBATCH --mem-per-gpu=17g
#SBATCH --gpus=4
#SBATCH --out=outfiles/df4.out
#SBATCH -t 144:00:00

source activate pytorch_p37
cd /home/ianpan/ufrc/deepfake/skp/
/home/ianpan/anaconda3/envs/pytorch_p37/bin/python run.py configs/experiments/experiment026.yaml train --gpu 0,1,2,3 --num-workers 4 
