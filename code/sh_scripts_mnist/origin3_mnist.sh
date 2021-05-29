#!/bin/bash

ratio_list_scale="0.01 0.02 0.03 0.04 0.05"
for ratio3 in $ratio_list_scale
do
	python -u main_origin_mnist.py --gpu 0 --seed 32 --scale --epoch 100 --noniid 0.4 --attack_mode="poison" --attack_ratio ${ratio3} --test_label_acc --target_random --dataset="mnist" --model_path="./mnist/origin/noniid_0.4/ratio_${ratio3}/poison_${ratio3}_Scale_0.pt" 2>&1 | tee ./mnist/origin/noniid_0.4/ratio_${ratio3}/poison_${ratio3}_Scale_0.txt

done
