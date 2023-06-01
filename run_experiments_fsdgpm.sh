#!/bin/bash

SEED=0

PMNIST='--dataset mnist_permutations --samples_per_task 1000 --workers 4 --batch_size 10 --n_epochs 1 --glances 5 --mem_batch_size 300 --thres 0.99 --thres_add 0.0005 --cuda'
CIFAR='--dataset cifar100 --n_tasks 10 --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 50 --mem_batch_size 125 --thres 0.97 --thres_add 0.003 --cuda --freeze_bn --second_order --earlystop'
CSUPER='--dataset cifar100_superclass --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 50 --mem_batch_size 125 --thres 0.98 --thres_add 0.001 --cuda  --second_order --earlystop'
IMGNET='--data_path ../data/tiny-imagenet-200/ --dataset tinyimagenet --pc_valid 0.1 --loader class_incremental_loader --increment 5 --class_order random --workers 8 --batch_size 10 --test_batch_size 64 --n_epochs 10 --mem_batch_size 200 --thres 0.9 --thres_add 0.0025 --cuda'

FSDGPM='--model fsdgpm --inner_batches 2 --sharpness --method xdgpm --expt_name fsdgpm'
SAM='--model sam --sharpness --method xdgpm --expt_name sam'

## 10-Split CIFAR-100 DATASET
for seed in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=0 python main.py $CIFAR --seed $seed $FSDGPM --memories 1000 --lr 0.01 --eta1 0.001 --eta2 0.01  --data_path ../../../vinai/phinh2
done
## 20-Split CIFAR-100 SuperClass DATASET
for seed in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=0 python main.py $CSUPER --seed $SEED $FSDGPM --memories 1000 --lr 0.01 --eta1 0.01 --eta2 0.01 --data_path ../../../vinai/phinh2
done
## 40-Split TinyImageNet DATASET
# CUDA_VISIBLE_DEVICES=0 python main.py $CSUPER \
#                 --seed $SEED $FSDGPM \
#                 --model sam\
#                 --memories 1000 \
#                 --lr 0.01 \
#                 --eta1 0.01 \
#                 --eta2 0.01 \
#                 --expt_name fs-dgpm \
#                 --data_path "../../../vinai/phinh2/"




###### VISUALIZATION OF LOSS LANDSCAPE ######

#VISUAL='--visual_landscape --step_size 0.02 --dir_num 10'
#DATASET='--dataset pmnist --memories 1000 --batch_size 10 --n_epochs 10 --glances 1 --lr_factor 10 --lr_min 0.01 --lr_patience 1 --cuda'

# ER
##python main.py $DATASET $VISUAL --seed $SEED --model ER --lr 0.005 --expt_name er

# ER+FS
##python main.py $DATASET $VISUAL --seed $SEED --model ER --sharpness --inner_batches 1 --lr 0.001 --eta1 0.01 --expt_name fs-er