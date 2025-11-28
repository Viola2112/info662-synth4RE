#!/bin/bash
#SBATCH --job-name=gen_synth      ### Name of the job
#SBATCH --nodes=1                   ### Number of Nodes
#SBATCH --ntasks=1                  ### Number of Tasks
#SBATCH --cpus-per-task=1           ### Number of Tasks per CPU
#SBATCH --gres=gpu:1                ### Number of GPUs, 2 GPUs
#SBATCH --mem=120G                   ### Memory required, 85 GB
#SBATCH --partition=amperenodes     ### Cheaha Partition
#SBATCH --time=12:00:00             ### Estimated Time of Completion, 24 hour
#SBATCH --output=%x_%j.out          ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=%x_%j.err           ### Slurm Error file, %x is job name, %j is job id


module load CUDA/12.6.0
module load Anaconda3

conda activate generation_vijay2

# some preparation
python -m ipykernel install --user --name generation_vijay2 --display-name "Python (generation_vijay2)"
jupyter nbconvert --to notebook --inplace --ClearMetadataPreprocessor.enabled=True Generate_Synthetic.ipynb

# run the synthetic generation from command line
jupyter nbconvert --to markdown --execute Generate_Synthetic.ipynb --stdout
