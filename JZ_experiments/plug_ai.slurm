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
#SBATCH --account=sos@v100
#SBATCH -C v100-16g


## load Pytorch module
module purge

module load pytorch-gpu/py3/1.10.1

## launch script on every node
set -x

# code execution
srun python /gpfsdswork/projects/idris/sos/ssos023/Projects/Plug-AI/Plug-AI/JZ_experiments/main.py \
--dataset_dir /gpfsscratch/idris/sos/ssos022/Medical/Task01_BrainTumour/ \
--limit_sample 20 --batch_size 2 --lr 0.0001
