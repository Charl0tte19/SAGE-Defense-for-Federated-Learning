#!/bin/bash

ratio_list="0.01 0.02 0.03 0.04 0.05"

loc="/home/ubuntu/My_FL/code/mnist/seed_32/command"

seed="seed_32"

for i in $ratio_list
do
	c=$(cat ${loc}/0.4${i}com.txt)
	#cmd=$c"2>&1|tee ./mnist/noniid_0.4/ratio_"$i"/log.txt"
	$c 2>&1 | tee ./mnist/${seed}/noniid_0.4/ratio_${i}/log.txt
done
