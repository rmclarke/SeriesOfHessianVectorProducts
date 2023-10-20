#!/bin/bash

# arg1 options are fashion_mnist cifar-10 svhn
dataset=$1
dataset_options=("fashion_mnist" "cifar-10" "svhn" "cifar-100" "kin8nm" "uci_protein" "uci_energy")
if [[ ! " ${dataset_options[*]} " =~ " ${dataset} " ]]; then
    echo "Invalid dataset specified ${dataset}"
    exit
fi
# arg2 are HessianSeries FixedLR Adam KFAC
optimiser=$2
optimiser_options=("SGD" "Adam" "OursAdaptive" "OursDampedUnadaptive" "KFACDeepMind" "KFACKazuki" "AdamSimple")
if [[ ! " ${optimiser_options[*]} " =~ " ${optimiser} " ]]; then
    echo "Invalid optimiser specified ${optimiser}"
    exit
fi
time_s=${3:-1800}
runs_per_gpu=${4:-12}


echo "Running Ray with dataset: ${dataset}, optimiser: ${optimiser}, time: ${time_s} runs_per_gpu: ${runs_per_gpu}"
# Pass through some variables by just putting them at the start of the command
dataset=$dataset \
 optimiser=$optimiser \
 time_s=$time_s \
 runs_per_gpu=$runs_per_gpu \
 sbatch ray.slurm

# sbatch --output=${output_path} --error=${error_path} ray.slurm
