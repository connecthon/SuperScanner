#!/bin/bash
#SBATCH --job-name=connecthon_voxelnet_train_c1978241
#SBATCH --partition=cubric-dgx
#SBATCH --gpus=1
#SBATCH -o slurm_out/train%j.out
#SBATCH -e slurm_out/train%j.err
/home/c1978241/miniconda3/envs/flow/bin/python /cubric/data/c1978241/Connecthon_superscanner/SuperScanner/test.py
