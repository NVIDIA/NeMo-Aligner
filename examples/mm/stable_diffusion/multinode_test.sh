#!/bin/bash
#SBATCH -A coreai_dlalgo_genai
#SBATCH -p polar3
#SBATCH -N 2
#SBATCH -t 0:04:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --job-name=coreai_dlalgo_genai-draftp-mulitnode:*
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton

### Multinode setup
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export RDZV_ID=$RANDOM
export NNODES=4

echo Node IP: $head_node_ip

srun echo $head_node_ip:$RDZV_ID
