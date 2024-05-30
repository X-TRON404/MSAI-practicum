#!/bin/bash
#SBATCH --account=p32050
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=16G
#SBATCH --job-name=gpu_env_test
#SBATCH --output=./logs/train.out

/home/phv0465/.conda/envs/pytorch-1.11-py38/bin/python3 ./train.py