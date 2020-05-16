#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=8g
#SBATCH --out=outfiles/etl4.out
#SBATCH -t 144:00:00

source activate pytorch_p37
cd /home/ianpan/ufrc/deepfake/skp/etl
/home/ianpan/anaconda3/envs/pytorch_p37/bin/python compute_face_diffs.py --start 18782
