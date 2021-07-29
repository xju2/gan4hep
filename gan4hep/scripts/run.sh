#!/bin/bash
#SBATCH -J gan
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 4:00:00
#SBATCH -A m1759
#SBATCH -o %x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=xju@lbl.gov

./train.sh
