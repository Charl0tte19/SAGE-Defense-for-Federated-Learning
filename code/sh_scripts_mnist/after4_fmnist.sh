#!/bin/bash

ratio_list="0.01 0.02 0.03 0.04 0.05"

loc="/home/ubuntu/My_FL/code/fmnist/seed_32/command"

seed="seed_32"

for i in $ratio_list
do
 	c=$(cat ${loc}/0.8${i}com.txt)
	#cmd=$c" 2>&1 | tee ./shuffle_models/noniid_0.8/ratio_"$i"/log.txt"
	$c --gpu 1 2>&1 | tee ./fmnist/${seed}/noniid_0.8/ratio_${i}/log.txt
done
