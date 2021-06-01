#!/bin/bash

#SBATCH --output=R-a6-1n1c-%j.out
#SBATCH --partition=snowy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:0:00

my_home="/home/$USER"

module load foss/2019b
module load python/3.7.4

time mpirun python3 a6.py --grid_file $my_home/melbGrid.json \
--twitter_file $my_home/bigTwitter.json \
--score_file $my_home/AFINN.txt