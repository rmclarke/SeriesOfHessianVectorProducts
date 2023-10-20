#!/bin/bash

# arg1 options are fashion_mnist cifar-10 svhn
dataset=$1
dataset_options=("fashion_mnist" "cifar-10" "svhn")
if [[ ! " ${dataset_options[*]} " =~ " ${dataset} " ]]; then
    echo "Invalid dataset specified ${dataset}"
    exit
fi
# arg2 are HessianSeries FixedLR Adam KFAC
optimiser=$2
optimiser_options=("HessianSeries" "FixedLR" "Adam" "KFAC")
if [[ ! " ${optimiser_options[*]} " =~ " ${optimiser} " ]]; then
    echo "Invalid optimiser specified ${optimiser}"
    exit
fi
num_parallel_calls=${3:-12}

group_name="${dataset}_${optimiser}"

write_path="/rds/user/authorid/hpc-work/LearningSecondOrderOptimiser/runs/${group_name}"
# We've given up on specifying these for now
#output_path="/rds/user/authorid/hpc-work/LearningSecondOrderOptimiser/tune/${group_name}"
#error_path="/rds/user/authorid/hpc-work/LearningSecondOrderOptimiser/tune/${group_name}"

mkdir ${write_path}


echo "Running x50 optimal on ${dataset} with ${optimiser}"
# Pass through some variables by just putting them at the start of the command
dataset=$dataset \
 optimiser=$optimiser \
 num_parallel_calls=$num_parallel_calls \
 sbatch x50.slurm

# sbatch --output=${output_path} --error=${error_path} ray.slurm
