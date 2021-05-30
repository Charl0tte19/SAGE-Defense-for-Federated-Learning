#!/bin/bash

ratio_list="0.1 0.2 0.3 0.4 0.5"
ratio_list_scale="0.01 0.02 0.03 0.04 0.05"
seed="seed_12"
seed_num="12"
where="fmnist"
where_origin="fmnist_origin"
folder_noniid_04="notScale_0"
folder_noniid_04_scale="Scale_0"
folder_noniid_08="notScale_0"
folder_noniid_08_scale="Scale_0"

for ratio in $ratio_list
do
	python -u test_trained_models_fmnist.py --gpu 0 --seed ${seed_num} --target_random --test_label_acc --final_model="./${where}/${seed}/noniid_0.4/ratio_${ratio}/final.pt" 2>&1 | tee ./${where}/${seed}/noniid_0.4/ratio_${ratio}/testing.txt

	python -u test_trained_models_fmnist.py --gpu 0 --target_random --seed ${seed_num} --test_label_acc --final_model="./${where_origin}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04}.pt" 2>&1 | tee ./${where_origin}/${seed}/noniid_0.4/ratio_${ratio}/origin_testing.txt
	
done

for ratio in $ratio_list_scale
do
	python -u test_trained_models_fmnist.py --gpu 0 --seed ${seed_num} --target_random --test_label_acc --final_model="./${where}/${seed}/noniid_0.4/ratio_${ratio}/final.pt" 2>&1 | tee ./${where}/${seed}/noniid_0.4/ratio_${ratio}/testing.txt

	python -u test_trained_models_fmnist.py --gpu 0 --target_random --seed ${seed_num} --test_label_acc --final_model="./${where_origin}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04_scale}.pt" 2>&1 | tee ./${where_origin}/${seed}/noniid_0.4/ratio_${ratio}/origin_testing.txt
	
done

for ratio in $ratio_list
do
	python -u test_trained_models_fmnist.py --gpu 0 --seed ${seed_num} --target_random --test_label_acc --final_model="./${where}/${seed}/noniid_0.8/ratio_${ratio}/final.pt" 2>&1 | tee ./${where}/${seed}/noniid_0.8/ratio_${ratio}/testing.txt

	python -u test_trained_models_fmnist.py --gpu 0 --target_random --seed ${seed_num} --test_label_acc --final_model="./${where_origin}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08}.pt" 2>&1 | tee ./${where_origin}/${seed}/noniid_0.8/ratio_${ratio}/origin_testing.txt
		
done

for ratio in $ratio_list_scale
do
	python -u test_trained_models_fmnist.py --gpu 0 --seed ${seed_num} --target_random --test_label_acc --final_model="./${where}/${seed}/noniid_0.8/ratio_${ratio}/final.pt" 2>&1 | tee ./${where}/${seed}/noniid_0.8/ratio_${ratio}/testing.txt

	python -u test_trained_models_fmnist.py --gpu 0 --target_random --seed ${seed_num} --test_label_acc --final_model="./${where_origin}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08_scale}.pt" 2>&1 | tee ./${where_origin}/${seed}/noniid_0.8/ratio_${ratio}/origin_testing.txt
	
done	