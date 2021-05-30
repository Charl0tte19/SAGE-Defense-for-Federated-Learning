#!/bin/bash

ratio_list="0.1 0.2 0.3 0.4 0.5"

loc="/home/ubuntu/My_FL/code/mnist/seed_32/command"

seed="seed_32"

for i in $ratio_list
do
	c=$(cat ${loc}/0.8${i}com.txt)
	#cmd=$c"2>&1|tee ./mnist/noniid_0.8/ratio_"$i"/log.txt"
	$c 2>&1 | tee ./mnist/${seed}/noniid_0.8/ratio_${i}/log.txt
done


