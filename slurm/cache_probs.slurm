#!/bin/bash
#SBATCH --job-name=cache_probs # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --time 01:00:00         # total run time limit (HH:MM:SS)
#SBATCH --account=bcyi-delta-gpu # <- match to a "Project" returned by the "accounts" command
#SBATCH -o log/cache_probs/%j.out
module purge
module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load pytorch-conda  # ... or any appropriate modulesASADA>RE
conda activate doc_lm
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_LAUNCH_BLOCKING=1
echo "job is starting on `hostname`"
python cal_ppl.py --data data/penn --save trainedmodel/ptb/additional_finetuned --bptt 1000
for seed in 0 1 2 3; do
    echo "Running seed $seed"
    python cal_ppl.py --data data/penn --save trainedmodel/ptb_ensemble/seed"$seed"_additional_finetuned --bptt 1000
done