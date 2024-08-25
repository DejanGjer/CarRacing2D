#!/bin/bash
#SBATCH --job-name=car_race
#SBATCH --ntasks=1
#SBATCH --nodelist=n18
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --output slurm.%J.out
#SBATCH --error slurm.%J.err
#SBATCH --time=48:00:00

python main.py
