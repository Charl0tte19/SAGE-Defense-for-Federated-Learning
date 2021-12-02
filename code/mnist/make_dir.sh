#!/bin/bash
list="0.1 0.2 0.3 0.4 0.5 0.01 0.02 0.03 0.04 0.05"

mkdir noniid_0.4
for i in $list
do
	mkdir ./mnist/ratio_$i
done
mv r* ./mnist/noniid_0.4/
cp -r ./mnist/noniid_0.4 ./mnist/noniid_0.8
