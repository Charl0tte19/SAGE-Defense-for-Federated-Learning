#!/bin/bash

list="0.1 0.2 0.3 0.4 0.5 0.01 0.02 0.03 0.04 0.05"

for i in $list
do
	mkdir ./fmnist/ratio_$i
done

mkdir ./fmnist/noniid_0.4
mv ./fmnist/ra* ./fmnist/noniid_0.4/
cp -r ./fmnist/noniid_0.4 ./fmnist/noniid_0.8
