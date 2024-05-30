#!/bin/bash
#SBATCH --account=p32050
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=8G
#SBATCH --job-name=gpu_env_test
#SBATCH --output=./logs/info.out

/home/phv0465/.conda/envs/pytorch-1.11-py38/bin/python3 ./info.py