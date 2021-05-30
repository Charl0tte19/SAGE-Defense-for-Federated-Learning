ratio_list="0.1 0.2 0.3 0.4 0.5"
ratio_list_scale="0.01 0.02 0.03 0.04 0.05"
seed="seed_12"
where="fmnist"
folder_noniid_04="notScale_0"
folder_noniid_04_scale="Scale_0"
folder_noniid_08="notScale_0"
folder_noniid_08_scale="Scale_0"

for ratio in $ratio_list
do
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04}.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/a.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/log.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/b.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04}.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/c.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/log.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/d.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/log.txt | grep 'pretrained_model' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/pre.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04}.txt | grep 'are attackers with poison attack' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/num.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04}.txt | grep 'clients (' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/client.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04}.txt | grep 'Round' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/round.txt
done

for ratio in $ratio_list_scale
do
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04_scale}.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/a.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/log.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/b.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04_scale}.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/c.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/log.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/d.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/log.txt | grep 'pretrained_model' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/pre.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04_scale}.txt | grep 'are attackers with poison attack' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/num.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04_scale}.txt | grep 'clients (' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/client.txt
	cat ./${where}/${seed}/noniid_0.4/ratio_${ratio}/poison_${ratio}_${folder_noniid_04_scale}.txt | grep 'Round' > ./${where}/${seed}/noniid_0.4/ratio_${ratio}/round.txt
done

for ratio in $ratio_list
do
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08}.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/a.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/log.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/b.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08}.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/c.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/log.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/d.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/log.txt | grep 'pretrained_model' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/pre.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08}.txt | grep 'are attackers with poison attack' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/num.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08}.txt | grep 'clients (' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/client.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08}.txt | grep 'Round' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/round.txt
done

for ratio in $ratio_list_scale
do
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08_scale}.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/a.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/log.txt | grep 'Testing accuracy:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/b.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08_scale}.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/c.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/log.txt | grep 'Testing Label Acc:' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/d.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/log.txt | grep 'pretrained_model' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/pre.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08_scale}.txt | grep 'are attackers with poison attack' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/num.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08_scale}.txt | grep 'clients (' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/client.txt
	cat ./${where}/${seed}/noniid_0.8/ratio_${ratio}/poison_${ratio}_${folder_noniid_08_scale}.txt | grep 'Round' > ./${where}/${seed}/noniid_0.8/ratio_${ratio}/round.txt
done	