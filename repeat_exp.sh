#!/bin/bash

# arg1 is dataset
dataset=$1
dataset_options=("fashion_mnist" "cifar-10" "svhn" "uci_energy")
if [[ ! " ${dataset_options[*]} " =~ " ${dataset} " ]]; then
    echo "Invalid dataset specified ${dataset}"
    exit
fi

# arg2 is optimiser
optimiser=$2
optimiser_options=("SGD" "Adam" "KFACDeepMind" "OursAdaptive" "KFACDeepMindNoEMADecay" "OursUnadaptive" "SFNAdaptive" "SFNUnadaptive" "SFNAdaptiveNoEMADecay" "KFACKazuki" "OursDampedUnadaptive" "OursDampedUnadaptive3200" )
if [[ ! " ${optimiser_options[*]} " =~ " ${optimiser} " ]]; then
    echo "Invalid optimiser specified ${optimiser}"
    exit
fi
optimiser_path="./configs/${dataset}_optimal/${optimiser}.yaml"
dataset_path="./configs/${dataset}_optimal/${dataset}.yaml"

# arg3 is how many parallel calls to make at a time; defaults to 6 
num_parallel_calls=${3:-6}

group_name=${4:-jax_${dataset}_${optimiser}_graceful}
logdir=./runs/${group_name}
mkdir $logdir

declare -a seeds=("2119213981" "1608860012" "1021032354" "280853612" "1415121920" "503407898" "995043888" "333388907" "1971069637" "1335198443" "285161167" "894408494" "952170761" "704127742" "168220153" "48936849" "1822305184" "1550130155" "812730049" "833357148" "1043290698" "369867697" "1119789429" "495194068" "806185573" "980810461" "1323666201" "1112576223" "33383858" "735190115" "2114747825" "153301904" "1417633242" "572670284" "71283607" "545220238" "1708331336" "31319830" "795335164" "698059710" "1298677938" "1248108292" "129243081" "869963795" "1378116027" "73798405" "1729011228" "1539271366" "999822958" "1251819451")

echo "Running x50 optimal on ${dataset} with ${optimiser} using config ${optimiser_path}"
echo "Saving to group ${group_name}"
echo ${seeds[@]} | xargs -n 1 -P ${num_parallel_calls} python train.py -c ${dataset_path} ${optimiser_path} -l "${logdir}" -g "${groupdir}" -n "${optimiser}" --seed
