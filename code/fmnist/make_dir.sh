#!/bin/bash

list="0.1 0.2 0.3 0.4 0.5 0.01 0.02 0.03 0.04 0.05"

for i in $list
do
	mkdir ratio_$i
done

mkdir noniid_0.4
mv ra* noniid_0.4/
cp -r noniid_0.4 noniid_0.8
