#!/bin/bash -l
#$ -N ssae_train_0
#$ -l h_rt=6:00:00
#$ -l gpu=1
#$ -l gpu_type=L40S
#$ -pe omp 8
#$ -j y

module load miniconda
mamba activate identifiable

cd /projectnb/buinlp/amueller/identifiable_language/spadeFormalGrammars/

python train_on_text.py --config-name $1