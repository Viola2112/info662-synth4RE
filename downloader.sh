#!/bin/bash
#SBATCH --job-name=download_qwen3      ### Name of the job
#SBATCH --nodes=1                   ### Number of Nodes
#SBATCH --ntasks=1                  ### Number of Tasks
#SBATCH --cpus-per-task=1           ### Number of Tasks per CPU
#SBATCH --gres=gpu:1                ### Number of GPUs, 2 GPUs
#SBATCH --mem=120G                   ### Memory required, 85 GB
#SBATCH --partition=pascalnodes     ### Cheaha Partition
#SBATCH --time=01:00:00             ### Estimated Time of Completion, 1 hour
#SBATCH --output=%x_%j.out          ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=%x_%j.err           ### Slurm Error file, %x is job name, %j is job id


module load CUDA/12.6.0
module load Anaconda3

conda activate bert-gt

python downloader.py