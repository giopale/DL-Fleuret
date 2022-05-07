#!/bin/sh

#SBATCH --nodes 2
#SBATCH --ntasks-per-node 20
#SBATCH --cpus-per-task 1
#SBATCH --time 10:00:00
#SBATCH --output log
#SBATCH --error err
#SBATCH --partition 20cores

module load anaconda/2021.05/python-3.8

export XDG_RUNTIME_DIR=/home/palermo/DL-Fleuret/tmp/xdg_runtime_dir

python3 Network_MK1.py
