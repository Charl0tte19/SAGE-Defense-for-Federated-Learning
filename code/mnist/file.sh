#!/bin/bash
seed="seed_32"
hom="/home/ubuntu/My_FL/src/mnist/"$seed
c="/command"
noniid="0.4 0.8"
ratio_list="0.1 0.2 0.3 0.4 0.5"
ratio_list_scale="0.01 0.02 0.03 0.04 0.05"
rm temp.txt
#poison_0.1_notScale_0.txt
get_flag=0
for noniid in $noniid
do
    for ratio in $ratio_list
    do
    	path=noniid_${noniid}
        cd $hom/$path/ratio_${ratio}
        echo $hom/$path/ratio_${ratio}
        echo $noniid >> $hom/cur.txt
    	echo $ratio >> $hom/cur.txt
	#cat poison_${ratio}_notScale_0.txt | grep locals > local.txt
	#cat poison_${ratio}_notScale_0.txt | grep attacker: > attacker.txt
	find ./ -name poison_${ratio}_notScale_0.txt >> $hom/path.txt
	temp=$(cat $hom/path.txt)
	rm $hom/path.txt
	if [[ -n $temp ]];
	then
	    cat poison_${ratio}_notScale_0.txt | grep locals > $hom/local.txt
	    cat poison_${ratio}_notScale_0.txt | grep attacker: > $hom/attacker.txt
	    cp poison_${ratio}_notScale_0.txt $hom/poison.txt
	    #echo "1"
	else
	    cat poison_${ratio}_notScale_0.txt | grep locals > $hom/local.txt
	    cat poison_${ratio}_notScale_0.txt | grep attacker: > $hom/attacker.txt
	    cp poison_${ratio}_notScale_0.txt $hom/poison.txt
	    #echo "2"
	fi
	python $hom/temp.py
	cp $hom/localout.txt ./local.txt
	cp $hom/attackerout.txt ./attacker.txt
	com=$(cat $hom/com.txt)
	echo $com >> $hom$c/${noniid}${ratio}com.txt
	rm $hom/cur.txt
        #pwd >> $hom/temp.txt
    done
    for ratio in $ratio_list_scale
    do
    	path=noniid_${noniid}
        cd $hom/$path/ratio_${ratio}
        echo $hom/$path/ratio_${ratio}
        #cp poison_${ratio_lis}_notScale_0.txt ../../
        #cp poison_${ratio_lis}_notScale0.txt ../../
        find ./ -name poison_${ratio}_Scale_0.txt >> $hom/path.txt
        #cat $hom/path.txt
	temp=$(cat $hom/path.txt)
	rm $hom/path.txt
    	echo $noniid >> $hom/cur.txt
    	echo $ratio >> $hom/cur.txt
	if [[ -n $temp ]];
	then
	    cp poison_${ratio}_Scale_0.txt $hom/poison.txt
	    cat poison_${ratio}_Scale_0.txt | grep locals > $hom/local.txt
	    cat poison_${ratio}_Scale_0.txt | grep attacker: > $hom/attacker.txt
	    #echo "3"
	else
	    cp poison_${ratio}_Scale_0.txt $hom/poison.txt
	    cat poison_${ratio}_Scale_0.txt | grep locals > $hom/local.txt
	    cat poison_${ratio}_Scale_0.txt | grep attacker: > $hom/attacker.txt
	    #echo "4"
	fi
	python $hom/temp.py
	cp $hom/localout.txt ./local.txt
	cp $hom/attackerout.txt ./attacker.txt
	com=$(cat $hom/com.txt)
	echo $com >> $hom$c/${noniid}${ratio}com.txt
        #pwd >> $hom/temp.txt
	rm $hom/cur.txt
    done
done




#com=$(cat $hom/com.txt)
#$com

