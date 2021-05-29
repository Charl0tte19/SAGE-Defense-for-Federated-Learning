#!/bin/bash

ratio_list="0.1 0.2 0.3 0.4 0.5"
for ratio2 in $ratio_list
do
	python -u main_shuffle_mnist.py --gpu 0 --seed 32 --dataset="mnist" --epoch 20 --noniid 0.8 --attack_mode="poison" --attack_ratio ${ratio2} --test_label_acc --target_random --shuffle --model_path="./mnist/noniid_0.8/ratio_${ratio2}/poison_${ratio2}_notScale_0.pt" 2>&1 | tee ./mnist/noniid_0.8/ratio_${ratio2}/poison_${ratio2}_notScale_0.txt

done
