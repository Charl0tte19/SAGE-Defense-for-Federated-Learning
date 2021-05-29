#!/bin/bash

ratio_list="0.1 0.2 0.3 0.4 0.5"

loc="/home/ubuntu/My_FL/code/fmnist/seed_32/command"

seed="seed_32"

for i in $ratio_list
do
	c=$(cat ${loc}/0.4${i}com.txt)
	#cmd=$c"2>&1|tee ./shuffle_models/noniid_0.4/ratio_"$i"/log.txt"
	$c 2>&1 | tee ./fmnist/${seed}/noniid_0.4/ratio_${i}/log.txt
done
