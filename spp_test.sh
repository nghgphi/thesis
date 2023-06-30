#!/bin/bash

SEED=0

# arguments for dataset
PMNIST='--dataset mnist_permutations --memories 200 --samples_per_task 1000 --workers 4 --batch_size 10 --n_epochs 1 --glances 5 --mem_batch_size 300 --thres 0.99 --thres_add 0.0005 --cuda --earlystop'
CIFAR='--dataset cifar100 --n_tasks 10 --memories 1000 --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 50 --mem_batch_size 125 --thres 0.97 --thres_add 0.003 --cuda --freeze_bn --second_order --earlystop'
CSUPER='--dataset cifar100_superclass --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 50 --mem_batch_size 125 --thres 0.98 --thres_add 0.001 --cuda  --second_order --earlystop'
IMGNET='--data_path ../data/tiny-imagenet-200/ --dataset tinyimagenet --pc_valid 0.1 --loader class_incremental_loader --increment 5 --class_order random --workers 8 --batch_size 10 --test_batch_size 64 --n_epochs 10 --mem_batch_size 200 --thres 0.9 --thres_add 0.0025 --cuda'

# arguments for model
FSDGPM='--model fsdgpm --inner_batches 2 --sharpness --method xdgpm --expt_name fsdgpm'
SAM='--model sam --sharpness --method xdgpm --expt_name sam_at --rho 0.05'

# python main.py $PMNIST --seed $SEED $FSDGPM --memories 200 --lr 0.01 --eta1 0.05 --eta2 0.01 
echo 'run file main_new.py'
CUDA_VISIBLE_DEVICES=1 python main_new_at.py $CIFAR \
                            $SAM \
                            --seed $SEED \
                            --lr 0.05 \
                            --eta2 0.01 \
                            --lambda_at 0.09 \
                            --data_path ../

