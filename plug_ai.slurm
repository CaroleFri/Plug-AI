#!/bin/bash
#SBATCH --job-name=plug_ai
#SBATCH --output=plug_ai.out
#SBATCH --error=plug_ai.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=10
#SBATCH --account=ibu@v100
#SBATCH -C v100-16g


## load Pytorch module
module purge

module load pytorch-gpu/py3/1.10.1

## launch script on every node
set -x

# code execution
srun python plug_ai/main.py --config_file "./config_exemples/config_Brats.yaml"
