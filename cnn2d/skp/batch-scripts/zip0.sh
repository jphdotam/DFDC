#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=14g
#SBATCH --out=outfiles/zip0.out
#SBATCH -t 144:00:00

cd /home/ianpan/ufrc/deepfake/data/dfdc
tar -czvf diffs2.tar.gz diffs/